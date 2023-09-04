# Model script for GNN embedding module
# ---------------------------------------------
#
#
from collections import namedtuple

import dgl    # https://github.com/dmlc/dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from DKG import settings
from DKG.model.gnn import RGCN, KGTransformer
from DKG.model.tpp import LogNormMixTPP
from DKG.utils.model_utils import node_norm_to_edge_norm, get_embedding

MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])



class EmbeddingUpdater(nn.Module):
    """
    A module that updates the embeddings of dynamic entities and relations in a knowledge graph.

    Args:
        num_nodes (int): The total number of nodes in the knowledge graph.
        in_dim (int): The input dimension of the embeddings.
        structural_hid_dim (int): The hidden dimension of the structural graph convolutional layer.
        temporal_hid_dim (int): The hidden dimension of the temporal graph convolutional layer.
        node_latest_event_time (torch.Tensor): A tensor containing the latest event time for each node in the graph.
        num_rels (int): The total number of relation types in the knowledge graph.
        rel_embed_dim (int): The dimension of the relation embeddings.
        graph_structural_conv (str): The type of structural graph convolutional layer to use.
        graph_temporal_conv (str): The type of temporal graph convolutional layer to use.
        num_gconv_layers (int): The number of graph convolutional layers to use.
        num_rnn_layers (int): The number of recurrent neural network layers to use.
        num_node_types (int): The number of node types in the knowledge graph.
        time_interval_transform (callable): A function that transforms time intervals.
        dropout (float): The dropout rate to use.
        activation (callable): The activation function to use.
        graph_name (str): The name of the knowledge graph.

    Attributes:
        num_nodes (int): The total number of nodes in the knowledge graph.
        num_rels (int): The total number of relation types in the knowledge graph.
        in_dim (int): The input dimension of the embeddings.
        structural_hid_dim (int): The hidden dimension of the structural graph convolutional layer.
        temporal_hid_dim (int): The hidden dimension of the temporal graph convolutional layer.
        node_latest_event_time (torch.Tensor): A tensor containing the latest event time for each node in the graph.
        graph_structural_conv (GraphStructuralRNNConv): The structural graph convolutional layer.
        graph_temporal_conv (GraphTemporalRNNConv): The temporal graph convolutional layer.
        structural_relation_rnn (RelationRNN): The relation RNN for structural embeddings.
        temporal_relation_rnn (RelationRNN): The relation RNN for temporal embeddings.
    """
    def __init__(self, num_nodes, in_dim, structural_hid_dim, temporal_hid_dim,
                 node_latest_event_time, num_rels, rel_embed_dim, 
                 graph_structural_conv='RGCN+RNN', graph_temporal_conv='RGCN+RNN',
                 num_gconv_layers=2, num_rnn_layers=1,
                 num_node_types=1, num_heads=8,
                 time_interval_transform=None, dropout=0.0, activation=None, graph_name=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.in_dim = in_dim
        self.structural_hid_dim = structural_hid_dim
        self.temporal_hid_dim = temporal_hid_dim
        self.node_latest_event_time = node_latest_event_time

        # Initialize the structural graph convolutional layer
        if graph_structural_conv in ['KGT+GRU','KGT+RNN', 'RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_structural_conv.split("+")
            self.graph_structural_conv = \
                GraphStructuralRNNConv(gconv, num_gconv_layers, rnn, num_rnn_layers, in_dim, structural_hid_dim,
                                       num_nodes, num_rels, rel_embed_dim,
                                       num_node_types=num_node_types,     # Meta-relation node type 
                                       num_heads=num_heads,     # Attention head
                                       dropout=dropout, activation=activation, graph_name=graph_name)
        elif graph_structural_conv is None:
            self.graph_structural_conv = None
        else:
            raise ValueError(f"Invalid graph structural conv: {graph_structural_conv}")

        # Initialize the temporal graph convolutional layer
        if graph_temporal_conv in ['KGT+GRU','KGT+RNN', 'RGCN+GRU', 'RGCN+RNN']:
            gconv, rnn = graph_temporal_conv.split("+")
            self.graph_temporal_conv = \
                GraphTemporalRNNConv(gconv, num_gconv_layers, rnn, num_rnn_layers, in_dim, temporal_hid_dim,
                                     node_latest_event_time, time_interval_transform, num_nodes, num_rels,
                                     num_node_types=num_node_types,   # Meta-relation node type 
                                     num_heads=num_heads,    # Attention head
                                     dropout=dropout, activation=activation, graph_name=graph_name)
        elif graph_temporal_conv is None:
            self.graph_temporal_conv = None
        else:
            raise ValueError(f"Invalid graph temporal conv: {graph_temporal_conv}")

        # Initialize the relation RNNs
        self.structural_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)
        self.temporal_relation_rnn = RelationRNN("RNN", num_rnn_layers, in_dim, rel_embed_dim, num_rels, dropout)

    def forward(self, prior_G, batch_G, cumul_G, static_entity_emb, dynamic_entity_emb,
                dynamic_relation_emb, device, batch_node_indices=None):
        """
        Update the embeddings of dynamic entities and relations in the knowledge graph.

        Args:
            prior_G (dgl.DGLGraph): The prior knowledge graph.
            batch_G (dgl.DGLGraph): The batch knowledge graph.
            cumul_G (dgl.DGLGraph): The cumulative knowledge graph.
            static_entity_emb (MultiAspectEmbedding): The static entity embeddings.
            dynamic_entity_emb (MultiAspectEmbedding): The dynamic entity embeddings.
            dynamic_relation_emb (MultiAspectEmbedding): The dynamic relation embeddings.
            device (torch.device): The device to use for computation.
            batch_node_indices (torch.Tensor): The indices of the nodes in the batch.

        Returns:
            MultiAspectEmbedding: The updated dynamic entity and relation embeddings.
        """
        assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), [emb.device for emb in dynamic_entity_emb]

        batch_G = batch_G.to(device)
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        # Compute the structural and temporal dynamic entity embeddings
        if self.graph_structural_conv is None:
            batch_structural_dynamic_entity_emb = None
        else:
            batch_structural_dynamic_entity_emb = \
                self.graph_structural_conv(batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices)
        if self.graph_temporal_conv is None:
            batch_temporal_dynamic_entity_emb = None
        else:
            batch_temporal_dynamic_entity_emb = \
                self.graph_temporal_conv(batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices)

        # Compute the structural and temporal dynamic relation embeddings
        batch_structural_dynamic_relation_emb = \
            self.structural_relation_rnn.forward(batch_G, dynamic_relation_emb.structural, static_entity_emb.structural, device)
        batch_temporal_dynamic_relation_emb = \
            self.temporal_relation_rnn.forward(batch_G, dynamic_relation_emb.temporal, static_entity_emb.temporal, device)

        # Update the dynamic entity embeddings
        updated_structural = dynamic_entity_emb.structural
        if batch_structural_dynamic_entity_emb is not None:
            updated_structural = dynamic_entity_emb.structural.clone()
            updated_structural[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_structural_dynamic_entity_emb.cpu()
        updated_temporal = dynamic_entity_emb.temporal
        if batch_temporal_dynamic_entity_emb is not None:
            updated_temporal = dynamic_entity_emb.temporal.clone()
            updated_temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_temporal_dynamic_entity_emb.cpu()
        updated_dynamic_entity_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        # Update the dynamic relation embeddings
        batch_G_rel = batch_G.edata['rel_type']
        batch_G_uniq_rel = torch.unique(batch_G_rel, sorted=True).long()

        updated_structural = dynamic_relation_emb.structural
        if batch_structural_dynamic_relation_emb is not None:
            updated_structural = dynamic_relation_emb.structural.clone()
            updated_structural[batch_G_uniq_rel] = batch_structural_dynamic_relation_emb.cpu()
        updated_temporal = dynamic_relation_emb.temporal
        if batch_temporal_dynamic_relation_emb is not None:
            updated_temporal = dynamic_relation_emb.temporal.clone()
            updated_temporal[batch_G_uniq_rel] = batch_temporal_dynamic_relation_emb.cpu()

        updated_dynamic_relation_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        return updated_dynamic_entity_emb, updated_dynamic_relation_emb
        updated_dynamic_relation_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        return updated_dynamic_entity_emb, updated_dynamic_relation_emb

# / Structural Embeddings
class GraphStructuralRNNConv(nn.Module):
    def __init__(self, graph_conv, num_gconv_layers, rnn, num_rnn_layers, in_dim, hid_dim, num_nodes, num_rels, rel_embed_dim,
                 add_entity_emb=False, dropout=0.2, num_node_types=1, num_heads=8, activation=None, graph_name=None):
        super().__init__()

        self.num_nodes = num_nodes  # num nodes in the entire G
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        self.num_rels = num_rels

        ### Node Encoder Layer
        if  graph_conv=='RGCN':
            self.graph_conv = RGCN(in_dim, hid_dim, hid_dim, n_layers=num_gconv_layers,
                                   num_rels=self.num_rels, regularizer="bdd",
                                   num_bases=50 if graph_name == "GDELT" else 100, dropout=dropout,
                                   activation=activation, layer_norm=False)
        elif graph_conv == 'KGT':
            self.graph_conv = KGTransformer(in_dim, hid_dim, self.num_heads, hid_dim,
                                            n_layers=num_gconv_layers,
                                            num_nodes=self.num_node_types,
                                            num_rels=self.num_rels,
                                            layer_norm=True)
        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        structural_rnn_in_dim = hid_dim
        self.add_entity_emb = add_entity_emb
        if self.add_entity_emb:
            structural_rnn_in_dim += hid_dim

        if rnn == "GRU":
            self.rnn_structural = nn.GRU(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                                         num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        elif rnn == "RNN":
            self.rnn_structural = nn.RNN(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                                         num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        """Structural RNN input"""
        batch_structural_static_entity_emb = static_entity_emb.structural[batch_G.ndata[dgl.NID].long()].to(device)
        if isinstance(self.graph_conv, RGCN):
            edge_norm = node_norm_to_edge_norm(batch_G)
            conv_structural_static_emb = self.graph_conv(batch_G, batch_structural_static_entity_emb, batch_G.edata['rel_type'].long(), edge_norm)
        
        # KGTransformer
        elif isinstance(self.graph_conv, KGTransformer):
            #edge_norm = node_norm_to_edge_norm(batch_G)
            conv_structural_static_emb = self.graph_conv(batch_G, batch_structural_static_entity_emb,
                                                         batch_G.ndata['node_type'].long(),   # num of node  
                                                         batch_G.edata['rel_type'].long()   # num of relationship
                                                         )
        else:
            conv_structural_static_emb = self.graph_conv(batch_G, batch_structural_static_entity_emb)  # shape=(# nodes in batch_G, dim-hidden)

        structural_rnn_input = [
            conv_structural_static_emb[batch_node_indices],
        ]
        if self.add_entity_emb:
            structural_rnn_input.append(static_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()].to(device))
        structural_rnn_input = torch.cat(structural_rnn_input, dim=1).unsqueeze(1)

        # Update structural dynamics
        structural_dynamic = dynamic_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()]
        structural_dynamic = structural_dynamic.to(device)

        output, hn = self.rnn_structural(structural_rnn_input, structural_dynamic.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_structural_dynamic_entity_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        return updated_structural_dynamic_entity_emb

    def extra_repr(self):
        field_desc = [f"add_entity_emb={self.add_entity_emb}"]
        return ", ".join(field_desc)

# / Temporal Embeddings
class GraphTemporalRNNConv(nn.Module):
    def __init__(self, graph_conv, num_gconv_layers, rnn, num_rnn_layers, in_dim, hid_dim,
                 node_latest_event_time, time_interval_transform, num_nodes, num_rels,
                 num_node_types = 1,
                 dropout=0.2, num_heads=8, activation=None, graph_name=None):
        super().__init__()

        self.num_nodes = num_nodes  # num nodes in the entire G
        self.num_node_types = num_node_types 
        self.num_heads = num_heads
        self.num_rels = num_rels

        ## Encoder module
        self.encoder_mode = graph_conv
        if graph_conv=='RGCN':
            self.graph_conv = RGCN(in_dim, hid_dim, hid_dim, n_layers=num_gconv_layers,
                                   num_rels=self.num_rels, regularizer="bdd",
                                   num_bases=50 if graph_name == "GDELT" else 100, dropout=dropout,
                                   activation=activation, layer_norm=False)
        elif graph_conv == 'KGT':
            self.graph_conv = KGTransformer(in_dim, hid_dim, self.num_heads, hid_dim, n_layers=num_gconv_layers,
                                            num_nodes=self.num_node_types,
                                            num_rels=self.num_rels,
                                            layer_norm=True)

        else:
            raise ValueError(f"Invalid graph conv: {graph_conv}")

        self.node_latest_event_time = node_latest_event_time
        self.time_interval_transform = time_interval_transform

        temporal_rnn_in_dim = hid_dim
        if rnn == "GRU":
            self.rnn_temporal = nn.GRU(input_size=temporal_rnn_in_dim, hidden_size=hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        elif rnn == "RNN":
            self.rnn_temporal = nn.RNN(input_size=temporal_rnn_in_dim, hidden_size=hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()  # update embeddings of all nodes in batch_G

        """Inter event times (in both directions)"""
        batch_G_sparse_inter_event_times = \
            EventTimeHelper.get_sparse_inter_event_times(batch_G, self.node_latest_event_time[..., 0])
        EventTimeHelper.get_inter_event_times(batch_G, self.node_latest_event_time[..., 0], update_latest_event_time=True)

        rev_batch_G = dgl.reverse(batch_G, copy_ndata=True, copy_edata=True)
        rev_batch_G.num_relations = batch_G.num_relations
        rev_batch_G.num_all_nodes = batch_G.num_all_nodes
        rev_batch_G_sparse_inter_event_times = \
            EventTimeHelper.get_sparse_inter_event_times(rev_batch_G, self.node_latest_event_time[..., 1])
        EventTimeHelper.get_inter_event_times(rev_batch_G, self.node_latest_event_time[..., 1], update_latest_event_time=True)

        """Temporal RNN input"""
        batch_temporal_static_entity_emb = static_entity_emb.temporal[batch_G.ndata[dgl.NID].long()].to(device)
        edge_norm = (1 / self.time_interval_transform(batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)
        
        # Encoding dimension
        #  !norm is time decay factor
        if isinstance(self.graph_conv, RGCN):
            batch_G_conv_temporal_static_emb = self.graph_conv(batch_G, batch_temporal_static_entity_emb,
                                                            batch_G.edata['rel_type'].long(), edge_norm)
        elif isinstance(self.graph_conv, KGTransformer):
            batch_G_conv_temporal_static_emb = self.graph_conv(batch_G, batch_temporal_static_entity_emb,
                                                               batch_G.ndata['node_type'].long(),   # num ofnode  
                                                               batch_G.edata['rel_type'].long(),   # num of relationship
                                                               norm=None
                                                            )
        else:
            raise ValueError(f" <-- Invalid graph conv")

        temporal_rnn_input_batch_G = torch.cat([
            batch_G_conv_temporal_static_emb,
        ], dim=1)[batch_node_indices].unsqueeze(1)

        rev_batch_temporal_static_entity_emb = static_entity_emb.temporal[rev_batch_G.ndata[dgl.NID].long()].to(device)
        rev_edge_norm = (1 / self.time_interval_transform(rev_batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(max=10.0)

        # Encoding node dimension
        if isinstance(self.graph_conv, RGCN):
            rev_batch_G_conv_temporal_static_emb = self.graph_conv(rev_batch_G, rev_batch_temporal_static_entity_emb, batch_G.edata['rel_type'].long(),
                                                                   rev_edge_norm)
        elif isinstance(self.graph_conv, KGTransformer):
            rev_batch_G_conv_temporal_static_emb = self.graph_conv(rev_batch_G, rev_batch_temporal_static_entity_emb, 
                                                                   rev_batch_G.ndata['node_type'].long(),   # num ofnode
                                                                   rev_batch_G.edata['rel_type'].long())   # num of relationship   
        else:
            raise ValueError(f" <-- Invalid graph conv")

        temporal_rnn_input_rev_batch_G = torch.cat([
            rev_batch_G_conv_temporal_static_emb,
        ], dim=1)[batch_node_indices].unsqueeze(1)

        temporal_dynamic = dynamic_entity_emb.temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()].to(device)
        temporal_dynamic_batch_G = temporal_dynamic[..., 0]  # dynamics as a recipient
        temporal_dynamic_rev_batch_G = temporal_dynamic[..., 1]  # dynamics as a sender

        output, hn = self.rnn_temporal(temporal_rnn_input_batch_G, temporal_dynamic_batch_G.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_temporal_dynamic_batch_G = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        output, hn = self.rnn_temporal(temporal_rnn_input_rev_batch_G, temporal_dynamic_rev_batch_G.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_temporal_dynamic_rev_batch_G = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)

        updated_temporal_dynamic_entity_emb = torch.cat([updated_temporal_dynamic_batch_G.unsqueeze(-1),
                                                         updated_temporal_dynamic_rev_batch_G.unsqueeze(-1)], dim=-1)
        return updated_temporal_dynamic_entity_emb


class RelationRNN(nn.Module):
    def __init__(self, rnn, num_rnn_layers, rnn_in_dim, rnn_hid_dim, num_rels, dropout=0.0):
        super().__init__()
        if num_rnn_layers==1:
            dropout = 0.0

        # RNN layer
        if rnn == "GRU":
            self.rnn_relation = nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        elif rnn == "RNN":
            self.rnn_relation = nn.RNN(input_size=rnn_in_dim, hidden_size=rnn_hid_dim,
                                       num_layers=num_rnn_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Invalid rnn: {rnn}")

    def forward(self, batch_G, dynamic_relation_emb, static_entity_emb, device):
        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_rel = batch_G.edata['rel_type'].long()

        batch_G_src_nid = batch_G.ndata[dgl.NID][batch_G_src.long()].long()
        batch_G_dst_nid = batch_G.ndata[dgl.NID][batch_G_dst.long()].long()

        # aggregate entity embeddings by relation. transpose() is necessary to aggregate entity emb matrix row-wise.
        batch_G_src_emb_avg_by_rel_ = \
            scatter_mean(static_entity_emb[batch_G_src_nid].to(device).transpose(0, 1),
                         batch_G_rel).transpose(0, 1)  # shape=(max rel in batch_G, static entity emb dim)
        batch_G_dst_emb_avg_by_rel_ = \
            scatter_mean(static_entity_emb[batch_G_dst_nid].to(device).transpose(0, 1),
                         batch_G_rel).transpose(0, 1)  # shape=(max rel in batch_G, static entity emb dim)

        # filter out relations that are non-existent in batch_G
        batch_G_uniq_rel = torch.unique(batch_G_rel, sorted=True)
        batch_G_src_emb_avg_by_rel = batch_G_src_emb_avg_by_rel_[batch_G_uniq_rel]  # shape=(# uniq rels in batch_G, static entity emb dim)
        batch_G_dst_emb_avg_by_rel = batch_G_dst_emb_avg_by_rel_[batch_G_uniq_rel]  # shape=(# uniq rels in batch_G, static entity emb dim)

        batch_G_dynamic_relation_emb = dynamic_relation_emb[batch_G_uniq_rel]
        batch_G_src_dynamic_relation_emb = batch_G_dynamic_relation_emb[..., 0].to(device)
        batch_G_dst_dynamic_relation_emb = batch_G_dynamic_relation_emb[..., 1].to(device)

        output, hn = self.rnn_relation(batch_G_src_emb_avg_by_rel.unsqueeze(1),
                                       batch_G_src_dynamic_relation_emb.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_batch_G_src_dynamic_relation_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)
        output, hn = self.rnn_relation(batch_G_dst_emb_avg_by_rel.unsqueeze(1),
                                       batch_G_dst_dynamic_relation_emb.transpose(0, 1).contiguous())  # transpose to make shape to be (num_layers, batch, hidden_size)
        updated_batch_G_dst_dynamic_relation_emb = hn.transpose(0, 1)  # transpose to make shape to be (batch, num_layers, hidden_size)
        updated_batch_G_dynamic_relation_emb = torch.cat([updated_batch_G_src_dynamic_relation_emb.unsqueeze(-1),
                                                          updated_batch_G_dst_dynamic_relation_emb.unsqueeze(-1)], dim=-1)

        return updated_batch_G_dynamic_relation_emb


class EventTimeHelper:
    @classmethod
    def get_sparse_inter_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_sparse_latest_event_times = cls.get_sparse_latest_event_times(batch_G, node_latest_event_time, _global)
        return batch_G.edata['time'] - batch_sparse_latest_event_times

    @classmethod
    def get_sparse_latest_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]

        batch_G_src, batch_G_dst = batch_G.edges()
        device = batch_G.ndata[dgl.NID].device
        if _global:
            return batch_latest_event_time[batch_G_dst.long(), -1].to(device)
        else:
            return batch_latest_event_time[batch_G_dst.long(), batch_G_nid[batch_G_src.long()]].to(device)

    @classmethod
    def get_inter_event_times(cls, batch_G, node_latest_event_time, update_latest_event_time=True):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]

        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_src, batch_G_dst = batch_G_src.long(), batch_G_dst.long()
        batch_G_time, batch_G_rel = batch_G.edata['time'], batch_G.edata['rel_type'].long()
        batch_G_time = batch_G_time.to(settings.INTER_EVENT_TIME_DTYPE)

        device = batch_G.ndata[dgl.NID].device
        batch_inter_event_times = torch.zeros(batch_G.num_nodes(), batch_G.num_all_nodes + 1, dtype=settings.INTER_EVENT_TIME_DTYPE).to(device)
        batch_inter_event_times[batch_G_dst, batch_G_nid[batch_G_src]] = \
            batch_G_time - batch_latest_event_time[batch_G_dst, batch_G_nid[batch_G_src]].to(device)

        batch_G.update_all(fn.copy_e('time', 't'), fn.max('t', 'max_event_time'))
        batch_G_max_event_time = batch_G.ndata['max_event_time'].to(settings.INTER_EVENT_TIME_DTYPE)

        batch_max_latest_event_time = batch_latest_event_time[:, -1].to(device)
        batch_G_max_event_time = torch.max(batch_G_max_event_time, batch_max_latest_event_time)
        batch_inter_event_times[:, -1] = batch_G_max_event_time - batch_max_latest_event_time

        if update_latest_event_time:
            node_latest_event_time[batch_G_nid[batch_G_dst], batch_G_nid[batch_G_src]] = batch_G_time.cpu()
            node_latest_event_time[batch_G_nid, -1] = batch_G_max_event_time.cpu()

        return batch_inter_event_times


### Static variation for RGCN
class StaticEmbeddingUpdater(nn.Module):
    def __init__(self, num_nodes, in_dim, structural_hid_dim, num_rels, rel_embed_dim, num_gconv_layers=2, dropout=0.0, activation=None, graph_name=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.in_dim = in_dim
        self.structural_hid_dim = structural_hid_dim

        # Initialize the structural graph convolutional layer as an RGCN layer from DGL
        self.graph_structural_conv = RGCN(in_dim, self.structural_hid_dim, self.structural_hid_dim,
                                          n_layers=num_gconv_layers,
                                          num_rels=self.num_rels, regularizer="bdd",
                                          num_bases=100,
                                          dropout=dropout,
                                          activation=activation, layer_norm=False)

    def forward(self, batch_G, entity_emb, device):
        batch_G = batch_G.to(device)
        node_features = {'node_type': entity_emb[batch_G.ndata[dgl.NID]].to(device)}
        h_dict = self.graph_structural_conv(batch_G, node_features)

        # Update entity embeddings
        updated_entity_emb = entity_emb.clone()
        updated_entity_emb[batch_G.ndata[dgl.NID]] = h_dict['node_type'].cpu()

        return updated_entity_emb

