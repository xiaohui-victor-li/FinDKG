import os
from collections import OrderedDict, defaultdict

import dgl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from DKG import settings   # Graph Configuraton
from DKG.model import DynamicGraphModel, StaticGraphModel, EventTimeHelper
from DKG.utils.eval_utils import RankingMetric, RegressionMetric
from DKG.utils.log_utils import logger, get_log_root_path


def evaluate(model: DynamicGraphModel, data_loader, entire_G, static_entity_emb,
             init_dynamic_entity_emb, init_dynamic_relation_emb, num_relations,
             args, phase, full_link_pred_eval, time_pred_eval, epoch=None, loss_weights=None):
    assert phase in ["Train", "Validation", "Test"], phase

    model.eval()
    torch.cuda.empty_cache()

    init_node_latest_event_time = model.node_latest_event_time.clone()
    dynamic_entity_emb = init_dynamic_entity_emb
    dynamic_relation_emb = init_dynamic_relation_emb
    eval_dict = {}
    log_msg = ""
    with torch.no_grad():
        if phase != "Train":
            eval_loss_dict = defaultdict(list)
            batch_tqdm = tqdm(data_loader)
            for i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(batch_tqdm):
                batch_tqdm.set_description(f"[{phase} / batch-{i}]")

                batch_loss_dict = compute_loss(model, args.eval, batch_G, static_entity_emb,
                                               dynamic_entity_emb, dynamic_relation_emb, args, batch_eid=None)
                for loss_term, loss_val in batch_loss_dict.items():
                    eval_loss_dict[loss_term].append(loss_val.item())

                dynamic_entity_emb, dynamic_relation_emb = \
                    model.embedding_updater.forward(prior_G, batch_G, cumul_G, static_entity_emb,
                                                    dynamic_entity_emb, dynamic_relation_emb, args.device)

            if epoch is not None:
                log_msg += f"[Epoch-{epoch}] "
            log_msg += f"{phase} loss total={sum([sum(l) for l in eval_loss_dict.values()]):.4f} | " \
                       f"{', '.join([f'{loss_term}={sum(loss_cumul):.4f}' for loss_term, loss_cumul in eval_loss_dict.items()])}"
            logger.info(log_msg)

        """Event time prediction"""
        if time_pred_eval:
            model.node_latest_event_time.copy_(init_node_latest_event_time)

            dynamic_entity_emb = init_dynamic_entity_emb
            dynamic_relation_emb = init_dynamic_relation_emb
            eval_time_diffs = []
            batch_tqdm = tqdm(data_loader)
            for i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(batch_tqdm):
                batch_tqdm.set_description(f"[{phase} / batch-{i}]")

                if phase in ["Test", "Validation"]:
                    batch_eval_dict = eval_time_prediction(model, batch_G, static_entity_emb,
                                                           dynamic_entity_emb, dynamic_relation_emb, args)
                    logger.info(f"[batch-{i}] MAE={batch_eval_dict['MAE']:.4f}, RMSE={batch_eval_dict['RMSE']:.4f}")
                    eval_time_diffs.extend(batch_eval_dict['time_diffs'])

                dynamic_entity_emb, dynamic_relation_emb = \
                    model.embedding_updater.forward(prior_G, batch_G, cumul_G, static_entity_emb,
                                                    dynamic_entity_emb, dynamic_relation_emb, args.device)

            eval_dict['MAE'] = RegressionMetric.mean_absolute_error(eval_time_diffs)
            eval_dict['RMSE'] = RegressionMetric.root_mean_squared_error(eval_time_diffs)
            logger.info(log_msg + f", MAE={eval_dict['MAE']:.4f}, RMSE={eval_dict['RMSE']:.4f}")

            if phase == "Test":
                log_root_path = get_log_root_path(args.graph, args.log_dir)
                with open(os.path.join(log_root_path, f"{args.result_file_prefix}{args.graph}_eval_{args.eval}_time_pred_test_result.txt"), 'w') as f:
                    f.write(f"{args.seed},{eval_dict['MAE']:.6f},{eval_dict['RMSE']:.6f}\n")

        """Temporal link prediction"""
        if full_link_pred_eval:
            model.node_latest_event_time.copy_(init_node_latest_event_time)

            dynamic_entity_emb = init_dynamic_entity_emb
            dynamic_relation_emb = init_dynamic_relation_emb
            eval_ranks_dict = None
            batch_tqdm = tqdm(data_loader)

            for i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(batch_tqdm):
                batch_tqdm.set_description(f"[{phase} / batch-{i}]")

                if phase == "Test" or \
                        (phase == "Validation" and args.graph == settings.GRAPH_ICEWS18 and i % 6 == 0) or \
                        (phase == "Validation" and args.graph == settings.GRAPH_ICEWS14 and i % 4 == 0) or \
                        (phase == "Validation" and args.graph == settings.GRAPH_GDELT and i % 20 == 0) or \
                        (phase == "Validation" and args.graph == settings.GRAPH_YAGO and i % 2 == 0) or \
                        (phase == "Validation" and args.graph == settings.GRAPH_WIKI and i % 3 == 0) or \
                        (phase == "Validation" and args.eval == 'edge'):

                    batch_eval_ranks_dict: OrderedDict = eval_link_prediction(
                        model, batch_G, cumul_G, entire_G, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb,
                        batch_times, args
                    )

                    batch_eval_edge_ranks = next(iter(batch_eval_ranks_dict.values()))
                    batch_eval_dict = {
                        'MRR': RankingMetric.mean_reciprocal_rank(batch_eval_edge_ranks),
                        'REC1': RankingMetric.recall(batch_eval_edge_ranks, 1),
                        'REC3': RankingMetric.recall(batch_eval_edge_ranks, 3),
                        'REC10': RankingMetric.recall(batch_eval_edge_ranks, 10),
                        'REC100': RankingMetric.recall(batch_eval_edge_ranks, 100),
                        'edge_ranks': batch_eval_edge_ranks,
                    }

                    #logger.info(f"[batch-{i}] MRR={batch_eval_dict['MRR']:.6f}, Rec@1={batch_eval_dict['REC1']:.4f}, "
                    #            f"Rec@3={batch_eval_dict['REC3']:.4f}, Rec@10={batch_eval_dict['REC10']:.4f}, "
                    #            f"Rec@100={batch_eval_dict['REC100']:.4f}")

                    if eval_ranks_dict is None:
                        eval_ranks_dict = batch_eval_ranks_dict
                    else:
                        assert eval_ranks_dict.keys() == batch_eval_ranks_dict.keys()
                        for k in eval_ranks_dict.keys():
                            eval_ranks_dict[k].extend(batch_eval_ranks_dict[k])

                dynamic_entity_emb, dynamic_relation_emb = \
                    model.embedding_updater.forward(prior_G, batch_G, cumul_G, static_entity_emb,
                                                    dynamic_entity_emb, dynamic_relation_emb, args.device)

            if loss_weights is None:
                weights_list = list(eval_ranks_dict.keys())
                mrr_list = [RankingMetric.mean_reciprocal_rank(eval_ranks) for eval_ranks in eval_ranks_dict.values()]
                max_mrr_idx, min_mrr_idx = np.argmax(mrr_list), np.argmin(mrr_list)
                best_weights, worst_weights = weights_list[max_mrr_idx], weights_list[min_mrr_idx]
                loss_weights = best_weights

            eval_ranks = eval_ranks_dict[loss_weights]
            eval_dict['MRR'] = RankingMetric.mean_reciprocal_rank(eval_ranks)
            for k in [1, 3, 10, 100]:
                eval_dict[f'REC{k}'] = RankingMetric.recall(eval_ranks, k)
            logger.info(log_msg + f", MRR={eval_dict['MRR']:.6f}, Rec@1={eval_dict['REC1']:.6f}, Rec@3={eval_dict['REC3']:.6f}, "
                                  f"Rec@10={eval_dict['REC10']:.6f}, Rec@100={eval_dict['REC100']:.6f}")

            if phase == "Test":
                log_root_path = get_log_root_path(args.graph, args.log_dir)
                with open(os.path.join(log_root_path, f"{args.result_file_prefix}{args.graph}_eval_{args.eval}_link_pred_test_result.txt"), 'w') as f:
                    f.write(f"{args.seed},{eval_dict['REC1']:.6f},{eval_dict['REC3']:.6f},{eval_dict['REC10']:.6f},{eval_dict['MRR']:.6f}\n")

        if phase != "Train":
            eval_dict['loss'] = sum([sum(l) for l in eval_loss_dict.values()])
        return eval_dict, dynamic_entity_emb, dynamic_relation_emb, loss_weights

# noinspection PyTypeChecker
def eval_link_prediction(model: DynamicGraphModel, batch_G, cumul_G, entire_G, static_entity_emb, dynamic_entity_emb,
                         dynamic_relation_emb, eval_times, args):
    """
    Evaluate the link prediction performance of a given dynamic graph model.

    Parameters:
    - model: The dynamic graph model to be evaluated.
    - batch_G: Graph batch.
    - cumul_G: Cumulative graph.
    - entire_G: Entire graph.
    - static_entity_emb: Static entity embeddings.
    - dynamic_entity_emb: Dynamic entity embeddings.
    - dynamic_relation_emb: Dynamic relation embeddings.
    - eval_times: Times at which evaluations are made.
    - args: Additional arguments.

    Returns:
    - eval_ranks_dict: A dictionary containing evaluation rankings.
    """
    # Ensure dynamic entity embeddings are on CPU device
    assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), [emb.device for emb in dynamic_entity_emb]
    model.eval()

    with torch.no_grad():
        cumul_G = cumul_G.to(args.device)

        # Extract evaluation time range
        eval_time_from, eval_time_to = eval_times[0], eval_times[-1]
        # # Filter edge IDs in the cumulative graph that fall within the evaluation time range
        eval_eid = torch.nonzero((eval_time_from <= cumul_G.edata['time']) & (cumul_G.edata['time'] <= eval_time_to), as_tuple=False).squeeze().view(-1)
        logger.debug(f"eval_eid: {eval_eid.shape}")

        assert len(dynamic_entity_emb.structural) == len(dynamic_entity_emb.structural) == len(static_entity_emb.structural) == len(static_entity_emb.temporal)

        with cumul_G.local_scope():
            """Compute edge likelihood"""
            edges_target_entity_log_prob = torch.empty(len(eval_eid), entire_G.num_nodes()).fill_(-1e20).cpu()  # shape: (# edges in cumul_G & belonging to eval_times, # nodes in the entire graph)
            if args.eval in ['edge', 'both']:
                cumul_G_structural_static_emb = static_entity_emb.structural[cumul_G.ndata[dgl.NID].long()].to(args.device)
                cumul_G_structural_dynamic_emb = dynamic_entity_emb.structural[cumul_G.ndata[dgl.NID].long()][:, -1, :].to(args.device)
                cumul_G_structural_combined_emb = model.combiner(cumul_G_structural_static_emb, cumul_G_structural_dynamic_emb, cumul_G)
                structural_dynamic_relation_emb = dynamic_relation_emb.structural[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn
                # Encoded embeddings for node
                cumul_G.ndata['emb'] = cumul_G_structural_combined_emb

                # Compute edge predictions
                _, edges_head_pred, edges_rel_pred, edges_tail_pred = \
                    model.edge_model(cumul_G, cumul_G_structural_combined_emb, cumul_G_structural_static_emb,
                                     cumul_G_structural_dynamic_emb, structural_dynamic_relation_emb,
                                     eid=eval_eid, return_pred=True)
                edges_target_entity_pred = edges_tail_pred
                edges_target_entity_log_prob = torch.log(torch.softmax(edges_target_entity_pred, dim=1)).cpu().detach()

            """Compute inter-event time likelihood"""
            edges_time_log_prob = torch.empty(len(eval_eid), entire_G.num_nodes()).fill_(-1e20).cpu()  # shape=(# edges in cumul_G & belonging to eval_times, # node in the entire graph)
            if args.eval in ['time', 'both']:
                cumul_G_temporal_static_emb = static_entity_emb.temporal[cumul_G.ndata[dgl.NID].long()].to(args.device)
                cumul_G_temporal_dynamic_emb = dynamic_entity_emb.temporal[cumul_G.ndata[dgl.NID].long()][:, -1, :, :].to(args.device)
                temporal_dynamic_relation_emb = dynamic_relation_emb.temporal[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

                time_log_prob_eval_dict = \
                    eval_edges_iet(model, cumul_G,
                                   cumul_G_temporal_dynamic_emb, cumul_G_temporal_static_emb, temporal_dynamic_relation_emb,
                                   eval_times, entire_G, device=args.device,
                                   node_latest_event_time=model.node_latest_event_time,
                                   compute_ranking_metrics=False)
                edges_time_log_prob_cumul_G = time_log_prob_eval_dict['edge_scores']  # shape=(# edges in cumul_G & belonging to eval_times, # nodes in cumul_G)
                edges_time_log_prob[:, cumul_G.ndata[dgl.NID].cpu().detach().long()] = edges_time_log_prob_cumul_G  # ignore nodes that are not yet added to the graph

            """Evaluate triples"""
            entire_G_edges_src, entire_G_edges_dst = entire_G.edges()
            entire_G_edges_rel = entire_G.edata['rel_type']
            entire_G_edges = torch.cat((entire_G_edges_src.view(-1, 1), entire_G_edges_rel.view(-1, 1), entire_G_edges_dst.view(-1, 1)), dim=1).to(args.device)

            cumul_G_edges_src, cumul_G_edges_dst = cumul_G.edges()
            eval_triples = torch.cat((cumul_G.ndata[dgl.NID][cumul_G_edges_src[eval_eid].long()].view(-1, 1),
                                      cumul_G.edata['rel_type'][eval_eid].view(-1, 1),
                                      cumul_G.ndata[dgl.NID][cumul_G_edges_dst[eval_eid].long()].view(-1, 1)), dim=1).cpu()
            assert len(eval_triples) == len(edges_target_entity_log_prob) == len(edges_time_log_prob), (eval_triples.shape, edges_target_entity_log_prob.shape)

            batch_G_inter_event_times = EventTimeHelper.get_sparse_inter_event_times(batch_G, model.node_latest_event_time[..., 0])

            # weights for (target_entity_log_prob, time_log_prob)
            if args.eval == 'edge':
                log_prob_weights_list = [(1.0, 0.0)]
            elif args.eval == 'time':
                log_prob_weights_list = [(0.0, 1.0)]
            else:  # eval both
                time_log_prob_weights = np.linspace(1.0, 0.0, num=11).tolist() + [1.5, 2.0, 3.0]
                log_prob_weights_list = [(1.0, time_weight) for time_weight in time_log_prob_weights]
                assert log_prob_weights_list[0] == (1.0, 1.0)

            # eval_ranks_dict - key: weights, value: eval_ranks obtained with the corresponding weights
            eval_ranks_dict = OrderedDict([(weights, []) for weights in log_prob_weights_list])
            for (s, r, o), target_entity_log_prob, time_log_prob, inter_event_time in \
                    zip(eval_triples, edges_target_entity_log_prob, edges_time_log_prob, batch_G_inter_event_times):
                assert target_entity_log_prob.shape == time_log_prob.shape, (target_entity_log_prob.shape, time_log_prob.shape)

                target_ent = o
                for log_prob_weights in log_prob_weights_list:
                    target_entity_weight, time_weight = log_prob_weights
                    pred = target_entity_weight * target_entity_log_prob + time_weight * time_log_prob
                    pred_ground = pred[target_ent]
                    ob_pred_comp1 = (pred > pred_ground).data.cpu().numpy()
                    ob_pred_comp2 = (pred == pred_ground).data.cpu().numpy()
                    target_rank = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

                    eval_ranks_dict[log_prob_weights].append(target_rank)

            eval_ranks = eval_ranks_dict[log_prob_weights_list[0]]

        assert len(eval_ranks) == len(eval_eid), (len(eval_ranks), len(eval_eid))
        return eval_ranks_dict


def eval_time_prediction(model, batch_G, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, args):
    model.eval()

    with torch.no_grad():
        batch_G_temporal_static_emb = static_entity_emb.temporal[batch_G.ndata[dgl.NID].long()].to(args.device)
        batch_G_temporal_dynamic_emb = dynamic_entity_emb.temporal[batch_G.ndata[dgl.NID].long()][:, -1, :, :].to(args.device)
        temporal_dynamic_relation_emb = dynamic_relation_emb.temporal[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

        expected_event_time = model.inter_event_time_model.expected_event_time(
            batch_G, batch_G_temporal_dynamic_emb, batch_G_temporal_static_emb, temporal_dynamic_relation_emb,
            model.node_latest_event_time
        )  # shape=(# edges in batch_G,)

        true_event_time = batch_G.edata['time']
        time_diffs = true_event_time.view(-1) - expected_event_time.view(-1)
        time_diffs = time_diffs.cpu().detach().tolist()

        eval_dict = {
            'MAE': RegressionMetric.mean_absolute_error(time_diffs),
            'RMSE': RegressionMetric.root_mean_squared_error(time_diffs),
            'time_diffs': time_diffs,
        }
        return eval_dict

######## Staitc Implementation ########
def evaluate_static(model, data_loader, static_entity_emb, args, phase, full_link_pred_eval=True, loss_weights=(1.0, 0.0)):
    model.eval()
    eval_ranks_dict = None
    eval_dict = {}
    
    with torch.no_grad():
        if full_link_pred_eval:
            batch_tqdm = tqdm(data_loader)
            for i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(batch_tqdm):
                batch_tqdm.set_description(f"[{phase} / batch-{i}]")
                
                if phase in ["Test", "Validation"]:
                    # Replace with your actual function for link prediction evaluation
                    batch_eval_ranks_dict = eval_static_link_prediction(model, cumul_G, static_entity_emb, batch_times, args)
                    
                    batch_eval_edge_ranks = next(iter(batch_eval_ranks_dict.values()))
                    
                    batch_eval_dict = {
                        'MRR': RankingMetric.mean_reciprocal_rank(batch_eval_edge_ranks),
                        'REC1': RankingMetric.recall(batch_eval_edge_ranks, 1),
                        'REC3': RankingMetric.recall(batch_eval_edge_ranks, 3),
                        'REC10': RankingMetric.recall(batch_eval_edge_ranks, 10),
                        'REC100': RankingMetric.recall(batch_eval_edge_ranks, 100),
                        'edge_ranks': batch_eval_edge_ranks,
                    }
                    
                    if eval_ranks_dict is None:
                        eval_ranks_dict = batch_eval_ranks_dict
                    else:
                        for k in eval_ranks_dict.keys():
                            eval_ranks_dict[k].extend(batch_eval_ranks_dict[k])
                            
            eval_ranks = eval_ranks_dict[loss_weights]
            eval_dict['MRR'] = RankingMetric.mean_reciprocal_rank(eval_ranks)
            for k in [1, 3, 10, 100]:
                eval_dict[f'REC{k}'] = RankingMetric.recall(eval_ranks, k)

            # Log the results
            logger.info(f"{phase} Metrics: MRR={eval_dict['MRR']:.6f}, Rec@1={eval_dict['REC1']:.6f}, Rec@3={eval_dict['REC3']:.6f}, Rec@10={eval_dict['REC10']:.6f}, Rec@100={eval_dict['REC100']:.6f}")
            
            if phase == "Test":
                log_root_path = get_log_root_path(args.graph, args.log_dir)
                with open(os.path.join(log_root_path, f"{args.result_file_prefix}{args.graph}_eval_{args.eval}_static_link_pred_test_result.txt"), 'w') as f:
                    f.write(f"{args.seed},{eval_dict['REC1']:.6f},{eval_dict['REC3']:.6f},{eval_dict['REC10']:.6f},{eval_dict['MRR']:.6f}\n")
                
    return eval_dict, eval_ranks_dict

def eval_static_link_prediction(model, entire_G, static_entity_emb, eval_times, args):
    """
    Evaluate the link prediction performance of a given static graph model.
    Parameters:
    - model: The static graph model to be evaluated.
    - entire_G: Entire graph.
    - static_entity_emb: Static entity embeddings.
    - eval_times: Times at which evaluations are made.
    - args: Additional arguments.

    Returns:
    - eval_ranks_dict: A dictionary containing evaluation rankings.
    """
    model.eval()
    with torch.no_grad():
        # Ensure static entity embeddings are on CPU device
        assert static_entity_emb.device == torch.device('cpu')
        
        entire_G = entire_G.to(args.device)

        # Extract evaluation time range
        eval_time_from, eval_time_to = eval_times[0], eval_times[-1]
        eval_eid = torch.nonzero((eval_time_from <= entire_G.edata['time']) & (entire_G.edata['time'] <= eval_time_to), as_tuple=False).squeeze().view(-1)

        with entire_G.local_scope():
            # Compute edge likelihood
            edges_target_entity_log_prob = torch.empty(len(eval_eid), entire_G.num_nodes()).fill_(-1e20).cpu()
            entire_G_static_emb = static_entity_emb[entire_G.ndata[dgl.NID].long()].to(args.device)
            # Encoded embeddings for node
            entire_G.ndata['emb'] = entire_G_static_emb

            # Compute edge predictions
            edge_LL = model.edge_model(entire_G, entire_G_static_emb, eid=eval_eid)
            edges_target_entity_log_prob = torch.log(torch.softmax(-edge_LL, dim=1)).cpu().detach()

            # Evaluate triples
            entire_G_edges_src, entire_G_edges_dst = entire_G.edges()
            eval_triples = torch.cat((entire_G.ndata[dgl.NID][entire_G_edges_src[eval_eid].long()].view(-1, 1),
                                      entire_G.edata['rel_type'][eval_eid].view(-1, 1),
                                      entire_G.ndata[dgl.NID][entire_G_edges_dst[eval_eid].long()].view(-1, 1)), dim=1).cpu()
            eval_ranks = []
            for (s, r, o), target_entity_log_prob in zip(eval_triples, edges_target_entity_log_prob):
                target_ent = o
                pred = target_entity_log_prob
                pred_ground = pred[target_ent]
                ob_pred_comp1 = (pred > pred_ground).data.cpu().numpy()
                ob_pred_comp2 = (pred == pred_ground).data.cpu().numpy()
                target_rank = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                eval_ranks.append(target_rank)
            eval_ranks_dict = OrderedDict([((1.0, 0.0), eval_ranks)])

    return eval_ranks_dict


# noinspection PyTypeChecker
def eval_edges_iet(model, cumul_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb, eval_times,
                   entire_G, device, node_latest_event_time, compute_ranking_metrics=True):
    """
    Evaluate edges in cumul_G that belong to eval_times in terms of the corresponding log-prob of inter-event time
    """
    model.eval()
    eval_model = model.inter_event_time_model.log_prob_interval

    with torch.no_grad():
        cumul_G = cumul_G.to(device)

        time_from, time_to = eval_times[0], eval_times[-1]
        eval_eid = torch.nonzero((time_from <= cumul_G.edata['time']) & (cumul_G.edata['time'] <= time_to), as_tuple=False).squeeze().view(-1)
        logger.debug(f"eval_eid: {eval_eid.shape}")

        src, dst = cumul_G.edges()
        eval_src, eval_dst = src[eval_eid], dst[eval_eid]
        eval_rel = cumul_G.edata['rel_type'][eval_eid]
        eval_edge_time = cumul_G.edata['time'][eval_eid]
        entire_G = entire_G.to(device)

        evaluator = EdgeEvaluator(
            eval_model=eval_model,
            eval_G_num_relations=cumul_G.num_relations,
            time_interval=cumul_G.time_interval
        ).to(device)
        eval_edges = torch.cat((eval_src.view(-1, 1), eval_rel.view(-1, 1), eval_dst.view(-1, 1)), dim=1)

        eval_dict = evaluator.forward(eval_edges, eval_edge_time, cumul_G.nodes(), dynamic_entity_emb,
                                      static_entity_emb, dynamic_relation_emb, cumul_G.ndata[dgl.NID],
                                      entire_G.number_of_nodes(), node_latest_event_time)
        assert len(eval_dict['edge_ranks']) == len(eval_eid), (len(eval_dict['edge_ranks']), len(eval_eid))

        if compute_ranking_metrics:
            eval_dict.update({
                'MRR': RankingMetric.mean_reciprocal_rank(eval_dict['edge_ranks']),
                'REC1': RankingMetric.recall(eval_dict['edge_ranks'], 1),
                'REC3': RankingMetric.recall(eval_dict['edge_ranks'], 3),
                'REC10': RankingMetric.recall(eval_dict['edge_ranks'], 10),
                'REC100': RankingMetric.recall(eval_dict['edge_ranks'], 100),
            })

        return eval_dict


class EdgeEvaluator(nn.Module):
    def __init__(self, eval_model, eval_G_num_relations, time_interval):
        super().__init__()
        self.eval_model = eval_model
        self.num_relations = eval_G_num_relations
        self.time_interval = time_interval

    def forward(self, eval_edges, eval_edge_time, eval_G_nodes, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                eval_G_ndata_nid, entire_G_num_nodes, node_latest_event_time, perturb_dst=True):
        num_nodes = len(eval_G_nodes)

        if eval_G_nodes.device == torch.device('cpu'):
            MAX_EDGES = num_nodes * 10  # adjust appropriately according to the CPU memory size
        else:
            MAX_EDGES = 400000  # adjust appropriately according to the GPU memory size
        logger.debug(f"MAX_EDGES: {MAX_EDGES}")
        SPLIT_SIZE = max(1, int(MAX_EDGES / num_nodes))
        logger.debug(f"SPLIT_SIZE: {SPLIT_SIZE}")

        eval_edge_time = eval_edge_time.view(-1)
        assert len(eval_edges) == len(eval_edge_time), (len(eval_edges), len(eval_edge_time))
        src, rel, dst = eval_edges[:, 0].type(settings.DGL_GRAPH_ID_TYPE), eval_edges[:, 1], eval_edges[:, 2].type(settings.DGL_GRAPH_ID_TYPE)
        src_chunks = torch.split(src, SPLIT_SIZE)
        rel_chunks = torch.split(rel, SPLIT_SIZE)
        dst_chunks = torch.split(dst, SPLIT_SIZE)
        edge_time_chunks = torch.split(eval_edge_time, SPLIT_SIZE)
        logger.info(f"# chunks: {len(src_chunks)}")

        edge_scores_all, edge_ranks_all = [], []
        for i, (eval_src_chunk, eval_rel_chunk, eval_dst_chunk, eval_edge_time_chunk) in \
                enumerate(zip(src_chunks, rel_chunks, dst_chunks, edge_time_chunks)):
            if perturb_dst:
                # evaluate the position of (src, dst) in [(src, dst')] for every possible dst'
                u = eval_src_chunk.repeat_interleave(num_nodes)
                v = eval_G_nodes.repeat(len(eval_src_chunk))
            else:
                # evaluate the position of (src', dst) in [(src', dst)] for every possible src'
                u = eval_G_nodes.repeat(len(eval_dst_chunk))
                v = eval_dst_chunk.repeat_interleave(num_nodes)

            eval_G = dgl.graph((u, v), num_nodes=num_nodes).to(eval_G_nodes.device)
            eval_G.edata['rel_type'] = eval_rel_chunk.repeat_interleave(num_nodes)
            eval_G.edata['time'] = eval_edge_time_chunk.repeat_interleave(num_nodes)
            eval_G.ndata[dgl.NID] = eval_G_ndata_nid
            eval_G.num_relations = self.num_relations
            eval_G.num_all_nodes = entire_G_num_nodes
            eval_G.time_interval = self.time_interval

            edge_scores = self.eval_model(eval_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                                          node_latest_event_time)

            assert len(eval_src_chunk) == len(eval_dst_chunk), (len(eval_src_chunk), len(eval_dst_chunk))
            edge_scores = edge_scores.view(len(eval_src_chunk), -1)  # [# src/dst entities in a chunk, # nodes in a graph]

            edge_scores_all.append(edge_scores.cpu().detach())

            if perturb_dst:
                true_edge_scores = edge_scores.gather(1, eval_dst_chunk.view(-1, 1).long())
            else:
                true_edge_scores = edge_scores.gather(1, eval_src_chunk.view(-1, 1).long())
            # noinspection PyUnresolvedReferences
            true_edge_ranks = ((edge_scores > true_edge_scores).sum(dim=1) + 1) + ((edge_scores == true_edge_scores).sum(dim=1) - 1.0) / 2
            true_edge_ranks = true_edge_ranks.cpu().detach().tolist()
            assert len(true_edge_ranks) == len(eval_src_chunk) == len(eval_dst_chunk), (len(true_edge_ranks), len(eval_src_chunk), len(eval_dst_chunk))

            edge_ranks_all.extend(true_edge_ranks)

        eval_dict = {
            'edge_scores': torch.cat(edge_scores_all, dim=0),  # tensor of shape (# edges, # nodes)
            'edge_ranks': edge_ranks_all  # list of length=# edges
        }
        assert eval_dict['edge_scores'].shape == (len(eval_edges), num_nodes), (eval_dict['edge_scores'].shape, (len(eval_edges), num_nodes))
        assert len(eval_dict['edge_ranks']) == len(eval_edges), (len(eval_dict['edge_ranks']), len(eval_edges))

        return eval_dict



def compute_loss(model, loss, batch_G, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, args, batch_eid=None):
    assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), [emb.device for emb in dynamic_entity_emb]

    if batch_eid is not None:
        assert len(batch_eid) > 0, batch_eid.shape
        sub_batch_G = dgl.edge_subgraph(batch_G, batch_eid.type(settings.DGL_GRAPH_ID_TYPE), preserve_nodes=False)
        sub_batch_G.ndata[dgl.NID] = batch_G.ndata[dgl.NID][sub_batch_G.ndata[dgl.NID].long()]  # map nid in sub_batch_G to nid in the full graph
        sub_batch_G = sub_batch_G.to(args.device)

        batch_eid = None  # this is needed to NOT perform further edge selection in the loss functions below
    else:
        sub_batch_G = batch_G.to(args.device)
    sub_batch_G.num_relations = batch_G.num_relations
    sub_batch_G.num_all_nodes = batch_G.num_all_nodes

    loss_dict = {}
    """Edge loss"""
    if loss in ['edge', 'both']:
        sub_batch_G_structural_static_entity_emb = static_entity_emb.structural[sub_batch_G.ndata[dgl.NID].long()].to(args.device)
        sub_batch_G_structural_dynamic_entity_emb = dynamic_entity_emb.structural[sub_batch_G.ndata[dgl.NID].long()][:, -1, :].to(args.device)  # [:, -1, :] to retrieve last hidden from rnn
        sub_batch_G_combined_emb = model.combiner(sub_batch_G_structural_static_entity_emb,
                                                  sub_batch_G_structural_dynamic_entity_emb,
                                                  sub_batch_G)
        structural_dynamic_relation_emb = dynamic_relation_emb.structural[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

        edge_LL = model.edge_model(sub_batch_G, sub_batch_G_combined_emb, eid=batch_eid,
                                   static_emb=sub_batch_G_structural_static_entity_emb,
                                   dynamic_emb=sub_batch_G_structural_dynamic_entity_emb,
                                   dynamic_relation_emb=structural_dynamic_relation_emb)
        loss_dict['edge'] = -edge_LL

    """Inter-event time loss"""
    if loss in ['time', 'both']:
        sub_batch_G_temporal_static_entity_emb = static_entity_emb.temporal[sub_batch_G.ndata[dgl.NID].long()].to(args.device)
        sub_batch_G_temporal_dynamic_entity_emb = dynamic_entity_emb.temporal[sub_batch_G.ndata[dgl.NID].long()][:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn
        temporal_dynamic_relation_emb = dynamic_relation_emb.temporal[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn
        inter_event_time_LL = model.inter_event_time_model.log_prob_density(
            sub_batch_G,
            sub_batch_G_temporal_dynamic_entity_emb,
            sub_batch_G_temporal_static_entity_emb,
            temporal_dynamic_relation_emb,
            model.node_latest_event_time,
            batch_eid,
            reduction='mean'
        )
        loss_dict['time'] = -inter_event_time_LL

    return loss_dict

