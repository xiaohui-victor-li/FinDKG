# Core Model Architecture for Dynamic Knowledge Graph
# --------------------------------------------------
# 
#
from collections import namedtuple

import numpy as np
import dgl    # https://github.com/dmlc/dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from DKG import settings
from DKG.model.embedding import EmbeddingUpdater, EventTimeHelper, StaticEmbeddingUpdater
from DKG.model.time_interval_transform import TimeIntervalTransform
from DKG.model.gnn import RGCN
from DKG.model.tpp import LogNormMixTPP
from DKG.utils.train_utils import nullable_string, activation_string
from DKG.utils.model_utils import node_norm_to_edge_norm, get_embedding

MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])

DKG_CONFIG_DICT = {
    'version': None,   # model version
    'result_file_prefix': None,  # prefix of the result file
    'graph': "YAGO",  # graph name to save the model
    'log_dir': 'DKG',  # log directory name
    'seed':101,  # random seed
    'num_relations': 15,  # number of relations
    'num_node_types': 12, # number of entity types, 1 by default for entity without type
    'epochs':100,  # number of training epochs
    'lr':0.001,    # learning rate
    'weight_decay': 0.00001,  # weight-decay
    'early_stop': True,   # whether to apply early stoping
    'patience': 10,    # number of times to be tolerate until early stopping
    'early_stop_criterion': 'MRR',  # evaluation criterion
    'eval': 'edge',
    'optimize': 'edge',   # 'edge' or 'both'
    'clean_up_run_best_checkpoint': False, 
    'eval_every': 1,   # perform evaluation every k epoch(s)
    'eval_from': 0,  
    'full_link_pred_validation': True,
    'full_link_pred_test': True, 
    'time_pred_eval':False,
    'static_entity_embed_dim':200,
    'structural_dynamic_entity_embed_dim':200,
    'temporal_dynamic_entity_embed_dim':200,
    'inter_event_time_mode':'node2node_inter_event_times',
    'rel_embed_dim':200,
    'num_mix_components':128,
    'num_gconv_layers':2,
    'num_rnn_layers':1,
    'num_attn_heads':8,
    'rnn_truncate_every':100,
    'combiner_gconv':None, #  graph conv module for combiner
    'combiner_activation':activation_string('tanh'),  
    'static_dynamic_combine_mode':'concat',
    'dropout':0.2,
    'embedding_updater_structural_gconv':'RGCN+RNN',
    'embedding_updater_temporal_gconv':'RGCN+RNN',
    'embedding_updater_activation':activation_string('tanh'), 
    'inter_event_dtype': torch.float32,
    'gpu': -1  # -1 indicates CPU
}

class ConfigArgs:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

DKG_DEFAULT_CONFIG = ConfigArgs(DKG_CONFIG_DICT)


class DynamicGraphModel(nn.Module):
    def __init__(self, embedding_updater: EmbeddingUpdater, combiner, edge_model, inter_event_time_model, node_latest_event_time):
        super().__init__()
        self.embedding_updater = embedding_updater
        self.combiner = combiner
        self.edge_model = edge_model
        # Dynamic component: inter-event time model
        self.inter_event_time_model = inter_event_time_model
        self.node_latest_event_time = node_latest_event_time


class StaticGraphModel(nn.Module):
    def __init__(self, embedding_updater, edge_model):
        super().__init__()
        self.embedding_updater = embedding_updater
        self.edge_model = edge_model


# Product Graph Engine
class DynamicKGEngine:
    def __init__(self, args, num_nodes, model_type = 'KGT+RNN'):
        self.args = args
        self.model_type = model_type
        self.num_nodes = num_nodes
        self.num_relations = args.num_relations    # number of relations
        self.node_latest_event_time = torch.zeros(num_nodes,
                                                  num_nodes + 1,
                                                  2,
                                                  dtype=args.inter_event_dtype)
        self.time_interval_transform = TimeIntervalTransform(log_transform=True)
        self.device = args.device
        self.seed = args.seed

        # buildg up the model
        self.KG_model = self.build_dynamic_graph_model()

    # Internal function to build the model
    def init_embedding_updater(self):
        return EmbeddingUpdater(self.num_nodes ,
                                self.args.static_entity_embed_dim,
                                self.args.structural_dynamic_entity_embed_dim,
                                self.args.temporal_dynamic_entity_embed_dim,
                                self.node_latest_event_time,
                                self.num_relations,
                                self.args.rel_embed_dim,
                                num_node_types=  self.args.num_node_types,
                                graph_structural_conv=self.model_type,
                                graph_temporal_conv=self.model_type,
                                num_gconv_layers=self.args.num_gconv_layers,
                                num_rnn_layers=self.args.num_rnn_layers,
                                time_interval_transform=self.time_interval_transform,
                                dropout=self.args.dropout,
                                activation=self.args.embedding_updater_activation,
                                graph_name=self.args.graph).to(self.args.device)

    def init_embeddings(self):
        args = self.args
        static_entity_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_nodes, args.static_entity_embed_dim, zero_init=False),
            temporal=get_embedding(self.num_nodes, args.static_entity_embed_dim, zero_init=False),
        )

        init_dynamic_entity_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_nodes, [args.num_rnn_layers, args.structural_dynamic_entity_embed_dim], zero_init=True),
            temporal=get_embedding(self.num_nodes, [args.num_rnn_layers, args.temporal_dynamic_entity_embed_dim, 2], zero_init=True),
        )

        init_dynamic_relation_embeds = MultiAspectEmbedding(
            structural=get_embedding(self.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
            temporal=get_embedding(self.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
        )

        return static_entity_embeds, init_dynamic_entity_embeds, init_dynamic_relation_embeds


    def build_dynamic_graph_model(self):
        """ Build the dynamic graph model
        """
        args = self.args
        # Freeze the random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device=='cuda':
            torch.cuda.manual_seed_all(args.seed)

        self.embedding_updater = self.init_embedding_updater()

        self.combiner = Combiner(args.static_entity_embed_dim,
                                args.structural_dynamic_entity_embed_dim,
                                args.static_dynamic_combine_mode,
                                args.combiner_gconv,
                                self.num_relations,
                                args.dropout,
                                args.combiner_activation).to(self.device)

        # Temporal Triplet prediction model
        self.edge_model = EdgeModel(self.num_nodes,
                                    self.num_relations,
                                    self.args.rel_embed_dim,
                                    self.combiner,
                                    dropout=self.args.dropout).to(self.device)

        self.inter_event_time_model = InterEventTimeModel(dynamic_entity_embed_dim=self.args.temporal_dynamic_entity_embed_dim,
                                                          static_entity_embed_dim=self.args.static_entity_embed_dim,
                                                          num_rels=self.num_relations,
                                                          rel_embed_dim=self.args.rel_embed_dim, 
                                                          num_mix_components=self.args.num_mix_components,
                                                          time_interval_transform=self.time_interval_transform,
                                                          inter_event_time_mode=self.args.inter_event_time_mode,
                                                          dropout=self.args.dropout)

        return DynamicGraphModel(self.embedding_updater,
                                 self.combiner,
                                 self.edge_model,
                                 self.inter_event_time_model,
                                 self.node_latest_event_time).to(self.device)


# Core function to conduct temporal link prediction
def predict_link(model, cumul_G, tgt_heads, tgt_tails, relation_ids,
                 combined_emb, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, output_type='prob'):
    """
    Predicts the link probability for given heads, tails, and relation_ids.
    
    Args:
    - model: The trained model.
    - cumul_G: The cumulative graph.
    - tgt_heads: The target head nodes.
    - tgt_tails: The target tail nodes.
    - relation_ids: The relation IDs for the edges.
    - static_entity_emb: Static embeddings for entities.
    - dynamic_entity_emb: Dynamic embeddings for entities.
    - dynamic_relation_emb: Dynamic embeddings for relations.
    - eval_eid: Optional edge IDs for evaluation.
    - output_type: output prediction score of entity
    
    Returns:
    - edges_target_entity_log_prob: The log probability of predicted edges.
    """
    # Convert entire_G's node IDs to cumul_G's local IDs
    heads_local = [np.argwhere(cumul_G.ndata[dgl.NID].numpy() == x)[0][0] for x in tgt_heads]
    tails_local = [np.argwhere(cumul_G.ndata[dgl.NID].numpy() == y)[0][0] for y in tgt_tails]

    # Append the new edge for prediction
    cumul_G.add_edges(heads_local, tails_local)
    # store the relation_ids as edge data
    cumul_G.edata['rel_type'][-len(relation_ids):] = torch.tensor(relation_ids)
    
    edge_position_ids = np.arange(len(cumul_G.edata['rel_type']))[-len(tgt_heads):]
    pred_ids = torch.tensor(edge_position_ids, dtype=torch.int32)
    
    # Compute edge predictions
    _, edges_head_pred, edges_rel_pred, edges_tail_pred = model.edge_model(
        cumul_G, 
        combined_emb, 
        static_entity_emb,
        dynamic_entity_emb, 
        dynamic_relation_emb,
        eid=pred_ids, return_pred=True
    )
    if output_type=='prob':
        # tail entity probability
        pred_entity_prob = torch.softmax(edges_tail_pred, dim=1).cpu().detach().numpy()
    
    # tail entity likelihood
    return edges_tail_pred.cpu().detach().numpy()



class Combiner(nn.Module):
    def __init__(self, static_emb_dim, dynamic_emb_dim, static_dynamic_combine_mode,
                 graph_conv, num_rels=None, dropout=0.0, activation=None, num_gconv_layers=1):
        super().__init__()
        self.static_emb_dim = static_emb_dim
        self.dynamic_emb_dim = dynamic_emb_dim
        self.static_dynamic_combiner = StaticDynamicCombiner(static_dynamic_combine_mode, static_emb_dim, dynamic_emb_dim)

        self.num_gconv_layer = num_gconv_layers
        if graph_conv == RGCN.__name__:
            self.graph_conv_static = RGCN(self.static_emb_dim, self.static_emb_dim, self.static_emb_dim,
                                          n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                                          num_bases=100, dropout=dropout, activation=activation)
            self.graph_conv_dynamic = RGCN(self.dynamic_emb_dim, self.dynamic_emb_dim, self.dynamic_emb_dim,
                                           n_layers=num_gconv_layers, num_rels=num_rels, regularizer="bdd",
                                           num_bases=100, dropout=dropout, activation=activation)
        elif graph_conv is None:
            self.graph_conv_static = None
            self.graph_conv_dynamic = None
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        self.dropout = nn.Dropout(dropout)

    @property
    def combined_emb_dim(self):
        return self.static_dynamic_combiner.combined_emb_dim

    @classmethod
    def do_graph_conv(cls, G, emb, graph_conv):
        if graph_conv is None:
            return emb

        if isinstance(graph_conv, RGCN):
            edge_norm = node_norm_to_edge_norm(G)
            return graph_conv(G, emb, G.edata['rel_type'], edge_norm)
        else:
            return graph_conv(G, emb)

    def forward(self, static_emb, dynamic_emb, G=None):
        if self.static_dynamic_combiner.use_static_emb:
            static_emb = self.do_graph_conv(G, static_emb, self.graph_conv_static)
        if self.static_dynamic_combiner.use_dynamic_emb:
            dynamic_emb = self.do_graph_conv(G, dynamic_emb, self.graph_conv_dynamic)

        combined_emb = self.static_dynamic_combiner(self.dropout(static_emb), dynamic_emb)
        return combined_emb


class StaticDynamicCombiner(nn.Module):
    def __init__(self, mode, static_emb_dim, dynamic_emb_dim):
        super().__init__()
        self.mode = mode
        self.static_emb_dim = static_emb_dim
        self.dynamic_emb_dim = dynamic_emb_dim

        if self.mode == "concat":
            self.combined_emb_dim = static_emb_dim + dynamic_emb_dim
            self.use_static_emb = True
            self.use_dynamic_emb = True
        elif self.mode == "static_only":
            self.combined_emb_dim = static_emb_dim
            self.use_static_emb = True
            self.use_dynamic_emb = False
        elif self.mode == "dynamic_only":
            self.combined_emb_dim = dynamic_emb_dim
            self.use_static_emb = False
            self.use_dynamic_emb = True
        else:
            raise ValueError(f"Invalid combiner mode: {mode}")

    def forward(self, static_emb, dynamic_emb):
        if self.mode == "concat":
            return torch.cat([static_emb, dynamic_emb], dim=1)
        elif self.mode == "static_only":
            return static_emb
        elif self.mode == "dynamic_only":
            return dynamic_emb

    def __repr__(self):
        return "%s(mode=%s, static_emb_dim=%d, dynamic_emb_dim=%d, combined_emb_dim=%d)" % \
               (self.__class__.__name__, self.mode, self.static_emb_dim, self.dynamic_emb_dim, self.combined_emb_dim)


class GraphReadout(nn.Module):
    def __init__(self, combiner: Combiner, readout_op='max', readout_node_type="static"):
        super().__init__()

        self.combiner = combiner
        self.readout_node_type = readout_node_type
        if readout_node_type == "combined":
            self.node_emb_dim = self.combiner.combined_emb_dim
        elif readout_node_type == "static":
            self.node_emb_dim = self.combiner.static_emb_dim
        elif readout_node_type == "dynamic":
            self.node_emb_dim = self.combiner.dynamic_emb_dim
        else:
            raise ValueError(f"Invalid type: {readout_node_type}")

        self.readout_op = readout_op
        if readout_op in ['max', 'min', 'mean']:
            self.graph_emb_dim = self.node_emb_dim
        elif readout_op == 'weighted_sum':
            self.graph_emb_dim = 2 * self.node_emb_dim
            self.node_gating = nn.Sequential(
                nn.Linear(self.node_emb_dim, 1),
                nn.Sigmoid()
            )
            self.node_to_graph = nn.Linear(self.node_emb_dim, self.graph_emb_dim)
        else:
            raise ValueError(f"Invalid readout: {readout_op}")

    def forward(self, G, combined, static, dynamic):
        with G.local_scope():
            emb_dict = {"combined": combined, "static": static, "dynamic": dynamic}
            node_emb_name, node_emb = emb_dict[self.readout_node_type]

            if self.readout_op in ['max', 'min', 'mean']:
                if node_emb_name not in G.ndata:
                    G.ndata[node_emb_name] = node_emb
                return dgl.readout_nodes(G, node_emb_name, op=self.readout_op)
            elif self.readout_op == 'weighted_sum':
                return (self.node_gating(node_emb) * self.node_to_graph(node_emb)).sum(0, keepdim=True)
            else:
                raise ValueError(f"Invalid readout: {self.readout_op}")

# Dynamic Edge Model 
class EdgeModel(nn.Module):
    def __init__(self, num_entities, num_rels, rel_embed_dim, combiner, dropout=0.0, graph_readout_op='max'):
        super().__init__()

        self.num_entities = num_entities
        self.num_rels = num_rels
        assert isinstance(rel_embed_dim, int)
        self.rel_embed_dim = rel_embed_dim
        self.rel_embeds = get_embedding(num_rels, rel_embed_dim)
        self.combiner = combiner
        self.combined_emb_dim = combiner.combined_emb_dim
        self.graph_readout = GraphReadout(self.combiner, graph_readout_op)

        graph_emb_dim = self.graph_readout.graph_emb_dim
        self.transform_head = nn.Sequential(
            nn.Linear(graph_emb_dim, 4 * graph_emb_dim),
            nn.Tanh(),
            nn.Linear(4 * graph_emb_dim, num_entities)
        )

        node_graph_emb_dim = self.combined_emb_dim + self.graph_readout.graph_emb_dim
        self.transform_rel = nn.Sequential(
            nn.Linear(node_graph_emb_dim, node_graph_emb_dim),
            nn.Tanh(),
            nn.Linear(node_graph_emb_dim, self.num_rels)
        )

        node_graph_rel_emb_dim = self.combined_emb_dim + self.graph_readout.graph_emb_dim + rel_embed_dim * 2
        self.transform_tail = nn.Sequential(
            nn.Linear(node_graph_rel_emb_dim, 2 * node_graph_rel_emb_dim),
            nn.Tanh(),
            nn.Linear(2 * node_graph_rel_emb_dim, num_entities)
        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def log_prob_head(self, graph_emb, G, edge_head):
        emb = graph_emb.repeat(len(edge_head), 1)
        emb = self.dropout(emb)
        head_pred = self.transform_head(emb)

        return - self.criterion(head_pred, G.ndata[dgl.NID][edge_head.long()].long()), head_pred

    def log_prob_rel(self, edge_head_emb, graph_emb, edge_rels):
        graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)

        emb = torch.cat((edge_head_emb, graph_emb_repeat), dim=1)
        emb = self.dropout(emb)
        rel_pred = self.transform_rel(emb)

        return - self.criterion(rel_pred, edge_rels.long()), rel_pred

    def log_prob_tail(self, edge_head_emb, graph_emb, edge_rels, G, edge_tail, dynamic_relation_emb=None):
        graph_emb_repeat = graph_emb.repeat(len(edge_head_emb), 1)

        edge_static_rel_embeds = self.rel_embeds[edge_rels.long()]
        edge_dynamic_rel_embeds = dynamic_relation_emb[edge_rels.long()]
        edge_rel_embeds = torch.cat((edge_static_rel_embeds, edge_dynamic_rel_embeds), dim=1)

        emb = torch.cat((edge_head_emb, graph_emb_repeat, edge_rel_embeds), dim=1)
        emb = self.dropout(emb)
        tail_pred = self.transform_tail(emb)

        return - self.criterion(tail_pred, G.ndata[dgl.NID][edge_tail.long()].long()), tail_pred

    def graph_emb(self, G, combined_emb, static_emb, dynamic_emb):
        return self.graph_readout.forward(G, ('emb', combined_emb), ('static_emb', static_emb), ('dynamic_emb', dynamic_emb))

    def forward(self, G, combined_emb, static_emb, dynamic_emb, dynamic_relation_emb, eid=None, return_pred=False):
        with G.local_scope():
            G.ndata['emb'] = combined_emb

            edge_head, edge_tail = G.edges()
            edge_rel = G.edata['rel_type']
            if eid is not None:
                edge_head, edge_tail, edge_rel = edge_head[eid], edge_tail[eid], edge_rel[eid]

            # Node embedding
            edge_head_emb = G.ndata['emb'][edge_head.long()]  # [# edges, emb-dim]
            assert len(edge_head_emb.size()) == 2, edge_head_emb.size()

            dynamic_rel_emb = dynamic_relation_emb[:, :, 1]  # use (relation-dest) context
            graph_emb = self.graph_emb(G, combined_emb, static_emb, dynamic_emb)
            log_prob_tail, tail_pred = self.log_prob_tail(edge_head_emb, graph_emb, edge_rel, G, edge_tail,
                                                          dynamic_rel_emb)
            log_prob_rel, rel_pred = self.log_prob_rel(edge_head_emb, graph_emb, edge_rel)
            log_prob_head, head_pred = self.log_prob_head(graph_emb, G, edge_head)
            log_prob = log_prob_tail + 0.2 * log_prob_rel + 0.1 * log_prob_head

            if return_pred:
                return log_prob, head_pred, rel_pred, tail_pred
            else:
                return log_prob

# Static Edge Model 

class StaticEdgeModel(nn.Module):
    def __init__(self, num_entities, num_rels, rel_embed_dim, combiner, dropout=0.0):
        super().__init__()

        self.num_entities = num_entities
        self.num_rels = num_rels
        self.rel_embed_dim = rel_embed_dim
        self.rel_embeds = get_embedding(num_rels, rel_embed_dim)

        self.graph_readout = GraphReadout(combiner, readout_op='max', readout_node_type="static")
        graph_emb_dim = self.graph_readout.graph_emb_dim

        self.transform_head = nn.Sequential(
            nn.Linear(graph_emb_dim, 4 * graph_emb_dim),
            nn.Tanh(),
            nn.Linear(4 * graph_emb_dim, num_entities)
        )

        self.transform_rel = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.Tanh(),
            nn.Linear(graph_emb_dim, self.num_rels)
        )

        self.transform_tail = nn.Sequential(
            nn.Linear(graph_emb_dim + rel_embed_dim, 2 * (graph_emb_dim + rel_embed_dim)),
            nn.Tanh(),
            nn.Linear(2 * (graph_emb_dim + rel_embed_dim), num_entities)
        )

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def log_prob_head(self, graph_emb, G, edge_head):
        graph_emb = graph_emb.repeat(len(edge_head), 1)
        graph_emb = self.dropout(graph_emb)
        head_pred = self.transform_head(graph_emb)
        return - self.criterion(head_pred, G.ndata[dgl.NID][edge_head.long()].long()), head_pred

    def log_prob_rel(self, graph_emb, edge_rels):
        graph_emb = graph_emb.repeat(len(edge_rels), 1)
        graph_emb = self.dropout(graph_emb)
        rel_pred = self.transform_rel(graph_emb)
        return - self.criterion(rel_pred, edge_rels.long()), rel_pred

    def log_prob_tail(self, graph_emb, edge_rels, G, edge_tail):
        graph_emb = graph_emb.repeat(len(edge_rels), 1)
        rel_emb = self.rel_embeds[edge_rels.long()]
        combined_emb = torch.cat((graph_emb, rel_emb), dim=1)
        combined_emb = self.dropout(combined_emb)
        tail_pred = self.transform_tail(combined_emb)
        return - self.criterion(tail_pred, G.ndata[dgl.NID][edge_tail.long()].long()), tail_pred

    def forward(self, G, static_emb, eid=None, return_pred=False):
        with G.local_scope():
            edge_head, edge_tail = G.edges()
            edge_rel = G.edata['rel_type']
            if eid is not None:
                edge_head, edge_tail, edge_rel = edge_head[eid], edge_tail[eid], edge_rel[eid]

            graph_emb = self.graph_readout(G, None, static_emb, None)  # Assume that this works similarly to Dynamic Edge Model

            log_prob_tail, tail_pred = self.log_prob_tail(graph_emb, edge_rel, G, edge_tail)
            log_prob_rel, rel_pred = self.log_prob_rel(graph_emb, edge_rel)
            log_prob_head, head_pred = self.log_prob_head(graph_emb, G, edge_head)
            log_prob = log_prob_tail + 0.2 * log_prob_rel + 0.1 * log_prob_head

            if return_pred:
                return log_prob, head_pred, rel_pred, tail_pred
            else:
                return log_prob

class InterEventTimeModel(nn.Module):
    def __init__(self,
                 dynamic_entity_embed_dim,
                 static_entity_embed_dim,
                 num_rels,
                 rel_embed_dim,
                 num_mix_components,
                 time_interval_transform,
                 inter_event_time_mode,
                 mean_log_inter_event_time=0.0,
                 std_log_inter_event_time=1.0,
                 dropout=0.0):
        super().__init__()
        self.tpp_model = LogNormMixTPP(dynamic_entity_embed_dim, static_entity_embed_dim, num_rels, rel_embed_dim,
                                       inter_event_time_mode, num_mix_components, time_interval_transform,
                                       mean_log_inter_event_time, std_log_inter_event_time, dropout)

    def log_prob_density(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                         node_latest_event_time, batch_eid=None, reduction=None):
        return self.tpp_model.log_prob_density(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                               node_latest_event_time, batch_eid, reduction)

    def log_prob_interval(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                          node_latest_event_time, batch_eid=None, reduction=None):
        return self.tpp_model.log_prob_interval(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                                node_latest_event_time, batch_eid, reduction)

    def expected_event_time(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                            node_latest_event_time, batch_eid=None):
        return self.tpp_model.expected_event_time(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                                  node_latest_event_time, batch_eid)




