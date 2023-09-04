import argparse
import os
import pprint
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from DKG import settings, data
from DKG.eval import evaluate
from DKG.model import DynamicGraphModel, EmbeddingUpdater, Combiner, EdgeModel, InterEventTimeModel, MultiAspectEmbedding
from DKG.model.time_interval_transform import TimeIntervalTransform

import DKG.utils as utils
from DKG.utils.log_utils import add_logger_file_handler, get_log_root_path, logger
from DKG.utils.model_utils import get_embedding
from DKG.utils.train_utils import setup_cuda, EarlyStopping, nullable_string, activation_string


def main(args):
    """Data"""
    G = data.load_temporal_knowledge_graph(args.graph)
    num_relations = G.num_relations
    logger.info("\n" + "=" * 80 + "\n"
                f"[{args.graph}]\n"
                f"# nodes={G.number_of_nodes()}\n"
                f"# edges={G.number_of_edges()}\n"
                f"# relations={G.num_relations}\n" + "=" * 80 + "\n")

    collate_fn = partial(utils.collate_fn, G=G)
    train_data_loader = DataLoader(G.train_times, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(G.val_times, shuffle=False, collate_fn=collate_fn)
    test_data_loader = DataLoader(G.test_times, shuffle=False, collate_fn=collate_fn)

    """Model"""
    node_latest_event_time = torch.zeros(G.number_of_nodes(), G.number_of_nodes() + 1, 2, dtype=settings.INTER_EVENT_TIME_DTYPE)
    time_interval_transform = TimeIntervalTransform(log_transform=args.time_interval_log_transform)

    embedding_updater = EmbeddingUpdater(G.number_of_nodes(),
                                         args.static_entity_embed_dim,
                                         args.structural_dynamic_entity_embed_dim,
                                         args.temporal_dynamic_entity_embed_dim,
                                         args.embedding_updater_structural_gconv,
                                         args.embedding_updater_temporal_gconv,
                                         node_latest_event_time,
                                         G.num_relations,
                                         args.rel_embed_dim,
                                         num_gconv_layers=args.num_gconv_layers,
                                         num_rnn_layers=args.num_rnn_layers,
                                         time_interval_transform=time_interval_transform,
                                         dropout=args.dropout,
                                         activation=args.embedding_updater_activation,
                                         graph_name=args.graph).to(args.device)
    if args.static_dynamic_combine_mode == "static_only":
        assert args.embedding_updater_structural_gconv is None, args.embedding_updater_structural_gconv
        assert args.embedding_updater_temporal_gconv is None, args.embedding_updater_temporal_gconv

    combiner = Combiner(args.static_entity_embed_dim,
                        args.structural_dynamic_entity_embed_dim,
                        args.static_dynamic_combine_mode,
                        args.combiner_gconv,
                        G.num_relations,
                        args.dropout,
                        args.combiner_activation).to(args.device)

    edge_model = EdgeModel(G.number_of_nodes(),
                           G.num_relations,
                           args.rel_embed_dim,
                           combiner,
                           dropout=args.dropout).to(args.device)

    inter_event_time_model = InterEventTimeModel(dynamic_entity_embed_dim=args.temporal_dynamic_entity_embed_dim,
                                                 static_entity_embed_dim=args.static_entity_embed_dim,
                                                 num_rels=G.num_relations,
                                                 rel_embed_dim=args.rel_embed_dim,
                                                 num_mix_components=args.num_mix_components,
                                                 time_interval_transform=time_interval_transform,
                                                 inter_event_time_mode=args.inter_event_time_mode,
                                                 dropout=args.dropout)

    model = Model(embedding_updater, combiner, edge_model, inter_event_time_model, node_latest_event_time).to(args.device)

    """Static and dynamic entity embeddings"""
    static_entity_embeds = MultiAspectEmbedding(
        structural=get_embedding(G.num_nodes(), args.static_entity_embed_dim, zero_init=False),
        temporal=get_embedding(G.num_nodes(), args.static_entity_embed_dim, zero_init=False),
    )
    init_dynamic_entity_embeds = MultiAspectEmbedding(
        structural=get_embedding(G.num_nodes(), [args.num_rnn_layers, args.structural_dynamic_entity_embed_dim], zero_init=True),
        temporal=get_embedding(G.num_nodes(), [args.num_rnn_layers, args.temporal_dynamic_entity_embed_dim, 2], zero_init=True),
    )
    init_dynamic_relation_embeds = MultiAspectEmbedding(
        structural=get_embedding(G.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
        temporal=get_embedding(G.num_relations, [args.num_rnn_layers, args.rel_embed_dim, 2], zero_init=True),
    )

    """Set seed"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    log_root_path = get_log_root_path(args.graph, args.log_dir)
    overall_best_checkpoint_prefix = f"{args.graph}_evokg_overall_best_checkpoint"
    if args.best_checkpoint_prefix:
        overall_best_checkpoint_prefix = args.best_checkpoint_prefix
    run_best_checkpoint_prefix = f"{args.graph}_evokg_{args.exp_time}_run_best_checkpoint"
    if args.run_best_checkpoint_prefix:
        run_best_checkpoint_prefix = args.run_best_checkpoint_prefix
    stopper = EarlyStopping(args.graph, args.patience,
                            result_root=log_root_path,
                            run_best_checkpoint_prefix=run_best_checkpoint_prefix,
                            overall_best_checkpoint_prefix=overall_best_checkpoint_prefix, eval=args.eval)
    with open(os.path.join(log_root_path, f"{args.graph}_evokg_args_{args.exp_time}.txt"), 'w') as f:
        f.write(pprint.pformat(args.__dict__))

    params = list(model.parameters()) + [
        static_entity_embeds.structural, static_entity_embeds.temporal,
        init_dynamic_entity_embeds.structural, init_dynamic_entity_embeds.temporal,
        init_dynamic_relation_embeds.structural, init_dynamic_relation_embeds.temporal,
    ]
    edge_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    time_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    val_static_entity_emb = None  # static entity embeddings after processing up to validation data
    val_dynamic_entity_emb = None  # dynamic entity embeddings after processing up to validation data
    val_dynamic_relation_emb = None  # dynamic relation embeddings after processing up to validation data
    node_latest_event_time_post_valid = None  # node latest event time after processing up to validation data
    loss_weights = None

    if args.load_best:
        if os.path.isfile(args.load_best):
            overall_best_checkpoint_fpath = args.load_best
        else:
            overall_best_checkpoint_fpath = stopper.overall_best_checkpoint_fpath(args.load_best)

        if not os.path.exists(overall_best_checkpoint_fpath):
            logger.info(f"'--load-best {args.load_best}' ignored. No best model saved at {overall_best_checkpoint_fpath}.")
        else:
            best_checkpoint = stopper.load_checkpoint(overall_best_checkpoint_fpath)

            model, edge_optimizer, time_optimizer, static_entity_embeds, \
                init_dynamic_entity_embeds, val_dynamic_entity_emb, \
                init_dynamic_relation_embeds, val_dynamic_relation_emb, \
                node_latest_event_time_post_valid, loss_weights, best_args = \
                    unpack_checkpoint(best_checkpoint, model, edge_optimizer, time_optimizer)
            val_static_entity_emb = static_entity_embeds

            opt = "" if os.path.isfile(args.load_best) else f"opt={args.load_best}"
            best_score = "" if 'score' not in best_checkpoint else f"best score={best_checkpoint['score']:.4f}"
            desc = list(filter(None, [opt, best_score]))
            desc = f"({', '.join(desc)}) " if desc else ""
            logger.info(f"Loaded the best model {desc}from {overall_best_checkpoint_fpath}.")

    try:
        for epoch in range(args.epochs):
            """Training"""
            model.train()
            epoch_start_time = time.time()

            dynamic_entity_emb_post_train, dynamic_relation_emb_post_train = None, None

            model.node_latest_event_time.zero_()
            node_latest_event_time.zero_()
            dynamic_entity_emb = init_dynamic_entity_embeds
            dynamic_relation_emb = init_dynamic_relation_embeds

            num_train_batches = len(train_data_loader)
            train_tqdm = tqdm(train_data_loader)

            epoch_train_loss_dict = defaultdict(list)
            batch_train_loss = 0
            batches_train_loss_dict = defaultdict(list)

            for batch_i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(train_tqdm):
                train_tqdm.set_description(f"[Training / epoch-{epoch} / batch-{batch_i}]")
                last_batch = batch_i == num_train_batches - 1

                # Based on the current entity embeddings, predict edges in batch_G and compute training loss
                batch_train_loss_dict = compute_loss(model, args.optimize, batch_G, static_entity_embeds,
                                                     dynamic_entity_emb, dynamic_relation_emb, args)
                batch_train_loss += sum(batch_train_loss_dict.values())

                for loss_term, loss_val in batch_train_loss_dict.items():
                    epoch_train_loss_dict[loss_term].append(loss_val.item())
                    batches_train_loss_dict[loss_term].append(loss_val.item())

                if batch_i > 0 and ((batch_i % args.rnn_truncate_every == 0) or last_batch):
                    # noinspection PyUnresolvedReferences
                    batch_train_loss.backward()
                    batch_train_loss = 0

                    if args.optimize in ['edge', 'both']:
                        edge_optimizer.step()
                        edge_optimizer.zero_grad()
                    if args.optimize in ['time', 'both']:
                        time_optimizer.step()
                        time_optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    if args.embedding_updater_structural_gconv or args.embedding_updater_temporal_gconv:
                        for emb in dynamic_entity_emb + dynamic_relation_emb:
                            emb.detach_()

                    tqdm.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')} [Epoch {epoch:03d}-Batch {batch_i:03d}] "
                               f"batch train loss total={sum([sum(l) for l in batches_train_loss_dict.values()]):.4f} | "
                               f"{', '.join([f'{loss_term}={sum(loss_cumul):.4f}' for loss_term, loss_cumul in batches_train_loss_dict.items()])}")
                    batches_train_loss_dict = defaultdict(list)

                dynamic_entity_emb, dynamic_relation_emb = \
                    model.embedding_updater.forward(prior_G, batch_G, cumul_G, static_entity_embeds,
                                                    dynamic_entity_emb, dynamic_relation_emb, args.device)

                if last_batch:
                    dynamic_entity_emb_post_train = dynamic_entity_emb
                    dynamic_relation_emb_post_train = dynamic_relation_emb

            epoch_end_time = time.time()
            logger.info(f"[Epoch-{epoch}] Train loss total={sum([sum(l) for l in epoch_train_loss_dict.values()]):.4f} | "
                        f"{', '.join([f'{loss_term}={sum(loss_cumul):.4f}' for loss_term, loss_cumul in epoch_train_loss_dict.items()])} | "
                        f"elapsed time={epoch_end_time - epoch_start_time:.4f} secs")

            """Validation"""
            if epoch >= args.eval_from and epoch % args.eval_every == 0:
                dynamic_entity_emb = dynamic_entity_emb_post_train
                dynamic_relation_emb = dynamic_relation_emb_post_train

                val_dict, val_dynamic_entity_emb, val_dynamic_relation_emb, loss_weights = \
                    evaluate(model, val_data_loader, G, static_entity_embeds, dynamic_entity_emb, dynamic_relation_emb,
                             num_relations, args, "Validation", args.full_link_pred_validation, args.time_pred_eval, epoch)

                node_latest_event_time_post_valid = deepcopy(model.node_latest_event_time)

                if args.early_stop:
                    criterion = args.early_stop_criterion
                    if args.early_stop_criterion not in val_dict:
                        criterion = 'loss'

                    if criterion == 'MRR':
                        score = val_dict[criterion]
                    else:  # MAE, loss
                        score = -val_dict[criterion]
                    pack_args = (model, edge_optimizer, time_optimizer, static_entity_embeds,
                                 init_dynamic_entity_embeds, val_dynamic_entity_emb,
                                 init_dynamic_relation_embeds, val_dynamic_relation_emb,
                                 node_latest_event_time_post_valid, loss_weights, args)
                    if stopper.step(score, pack_checkpoint(*pack_args)):
                        logger.info(f"[Epoch-{epoch}] Early stop!")
                        break
    except KeyboardInterrupt:
        print("\n=== TRAINING INTERRUPTED! ===\n")
        print("Measuring the test performance using the best model found so far...")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        """Testing"""
        if args.early_stop:
            try:
                run_best_checkpoint = stopper.load_checkpoint()
                model, edge_optimizer, time_optimizer, val_static_entity_emb, \
                    init_dynamic_entity_embeds, val_dynamic_entity_emb, \
                    init_dynamic_relation_embeds, val_dynamic_relation_emb, \
                    node_latest_event_time_post_valid, loss_weights, _ = \
                        unpack_checkpoint(run_best_checkpoint, model, edge_optimizer, time_optimizer)
                logger.info("Loaded the best model so far for testing.")
            except Exception as e:
                logger.info(f"Failed to load the best model.\n{e}")

        if node_latest_event_time_post_valid is None:  # validation was not performed
            val_static_entity_emb = static_entity_embeds
            val_dynamic_entity_emb, val_dynamic_relation_emb = \
                forward_graphs(model, [train_data_loader, val_data_loader], static_entity_embeds,
                               init_dynamic_entity_embeds, init_dynamic_relation_embeds, args)
        else:
            assert val_static_entity_emb is not None
            assert val_dynamic_entity_emb is not None
            assert val_dynamic_relation_emb is not None
            model.node_latest_event_time.copy_(node_latest_event_time_post_valid)

        test_start_time = time.time()
        evaluate(model, test_data_loader, G, val_static_entity_emb, val_dynamic_entity_emb, val_dynamic_relation_emb,
                 num_relations, args, "Test", full_link_pred_eval=args.full_link_pred_test,
                 time_pred_eval=args.time_pred_eval, loss_weights=loss_weights)
        test_end_time = time.time()
        logger.info(f"Test elapsed time={test_end_time - test_start_time:.4f} secs")

        if args.clean_up_run_best_checkpoint:
            os.remove(stopper.run_best_checkpoint_fpath)
            logger.info(f"Removed the run best checkpoint ({stopper.run_best_checkpoint_fpath})")


def forward_graphs(model: DynamicGraphModel, data_loaders, static_entity_emb, init_dynamic_entity_emb,
                   init_dynamic_relation_emb, args):
    if not isinstance(data_loaders, list):
        data_loaders = list(data_loaders)

    with torch.no_grad():
        model.node_latest_event_time.zero_()
        dynamic_entity_emb = init_dynamic_entity_emb
        dynamic_relation_emb = init_dynamic_relation_emb

        for data_loader in data_loaders:
            data_loader_tqdm = tqdm(data_loader)
            for batch_i, (prior_G, batch_G, cumul_G, batch_times) in enumerate(data_loader_tqdm):
                data_loader_tqdm.set_description(f"[Forwarding: batch_times={batch_times[0]}-{batch_times[-1]}]")

                dynamic_entity_emb, dynamic_relation_emb = \
                    model.embedding_updater.forward(prior_G, batch_G, cumul_G, static_entity_emb,
                                                    dynamic_entity_emb, dynamic_relation_emb,
                                                    args.device, batch_node_indices=None)

        return dynamic_entity_emb, dynamic_relation_emb



def pack_checkpoint(model, edge_optimizer, time_optimizer, static_entity_emb,
                    init_dynamic_entity_emb, val_dynamic_entity_emb,
                    init_dynamic_relation_emb, val_dynamic_relation_emb,
                    node_latest_event_time, loss_weights, args):
    ckp = {
        'model': model.state_dict(),
        'edge_optimizer': edge_optimizer.state_dict(),
        'time_optimizer': time_optimizer.state_dict(),
        'static_entity_emb': static_entity_emb,
        'init_dynamic_entity_emb': init_dynamic_entity_emb,
        'val_dynamic_entity_emb': val_dynamic_entity_emb,
        'init_dynamic_relation_emb': init_dynamic_relation_emb,
        'val_dynamic_relation_emb': val_dynamic_relation_emb,
        'node_latest_event_time': node_latest_event_time.to_sparse(),  # this is node_latest_event_time_post_valid
        'loss_weights': loss_weights,
        'args': args,
    }
    return ckp


def unpack_checkpoint(checkpoint, model, edge_optimizer=None, time_optimizer=None):
    model.load_state_dict(checkpoint['model'], strict=False)
    try:
        if edge_optimizer:
            edge_optimizer.load_state_dict(checkpoint['edge_optimizer'])
        if time_optimizer:
            time_optimizer.load_state_dict(checkpoint['time_optimizer'])
    except Exception:
        pass

    if checkpoint['node_latest_event_time'].is_sparse:
        node_latest_event_time = checkpoint['node_latest_event_time'].to_dense()
    else:
        node_latest_event_time = checkpoint['node_latest_event_time']

    return model, edge_optimizer, time_optimizer, checkpoint['static_entity_emb'], \
           checkpoint['init_dynamic_entity_emb'], checkpoint['val_dynamic_entity_emb'], \
           checkpoint['init_dynamic_relation_emb'], checkpoint['val_dynamic_relation_emb'], \
           node_latest_event_time, checkpoint['loss_weights'], checkpoint['args']


# Compute Loss for default dynamic KG model
def compute_loss(model, loss, batch_G, static_entity_emb, dynamic_entity_emb, dynamic_relation_emb, args, batch_eid=None):
    """
    Compute the loss for the given batch of data.

    Args:
        model (DKGModel): The DGL model to use for computing the loss.
        loss (str): The type of loss to compute. Can be 'edge', 'time', or 'both'.
        batch_G (DGLGraph): The batch of data to compute the loss for.
        static_entity_emb (StaticEntityEmb): The static entity embeddings.
        dynamic_entity_emb (DynamicEntityEmb): The dynamic entity embeddings.
        dynamic_relation_emb (DynamicRelationEmb): The dynamic relation embeddings.
        args (argparse.Namespace): The command-line arguments.
        batch_eid (torch.Tensor, optional): The batch of edge IDs to compute the loss for. Defaults to None.

    Returns:
        dict: A dictionary containing the computed loss for each type of loss.
    """
    # Ensure that all dynamic entity embeddings are on the CPU.
    assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), [emb.device for emb in dynamic_entity_emb]

    # If batch_eid is not None, create a subgraph of batch_G using only the specified edges.
    if batch_eid is not None:
        assert len(batch_eid) > 0, batch_eid.shape
        sub_batch_G = dgl.edge_subgraph(batch_G, batch_eid.type(settings.DGL_GRAPH_ID_TYPE), preserve_nodes=False)
        sub_batch_G.ndata[dgl.NID] = batch_G.ndata[dgl.NID][sub_batch_G.ndata[dgl.NID].long()]  # map nid in sub_batch_G to nid in the full graph
        sub_batch_G = sub_batch_G.to(args.device)

        # Set batch_eid to None to prevent further edge selection in the loss functions below.
        batch_eid = None
    else:
        sub_batch_G = batch_G.to(args.device)

    # Set the number of relations and all nodes in the subgraph.
    sub_batch_G.num_relations = batch_G.num_relations
    sub_batch_G.num_all_nodes = batch_G.num_all_nodes

    # Initialize the loss dictionary.
    loss_dict = {}

    """Edge loss"""
    if loss in ['edge', 'both']:
        # Get the static and dynamic entity embeddings for the subgraph.
        sub_batch_G_structural_static_entity_emb = static_entity_emb.structural[sub_batch_G.ndata[dgl.NID].long()].to(args.device)
        sub_batch_G_structural_dynamic_entity_emb = dynamic_entity_emb.structural[sub_batch_G.ndata[dgl.NID].long()][:, -1, :].to(args.device)  # [:, -1, :] to retrieve last hidden from rnn

        # Combine the static and dynamic entity embeddings for the subgraph.
        sub_batch_G_combined_emb = model.combiner(sub_batch_G_structural_static_entity_emb,
                                                  sub_batch_G_structural_dynamic_entity_emb,
                                                  sub_batch_G)

        # Get the dynamic relation embeddings for the subgraph.
        structural_dynamic_relation_emb = dynamic_relation_emb.structural[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

        # Compute the edge loss for the subgraph.
        edge_LL = model.edge_model(sub_batch_G, sub_batch_G_combined_emb, eid=batch_eid,
                                   static_emb=sub_batch_G_structural_static_entity_emb,
                                   dynamic_emb=sub_batch_G_structural_dynamic_entity_emb,
                                   dynamic_relation_emb=structural_dynamic_relation_emb)
        loss_dict['edge'] = -edge_LL

    """Inter-event time loss"""
    if loss in ['time', 'both']:
        # Get the static and dynamic entity embeddings for the subgraph.
        sub_batch_G_temporal_static_entity_emb = static_entity_emb.temporal[sub_batch_G.ndata[dgl.NID].long()].to(args.device)
        sub_batch_G_temporal_dynamic_entity_emb = dynamic_entity_emb.temporal[sub_batch_G.ndata[dgl.NID].long()][:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

        # Get the dynamic relation embeddings for the subgraph.
        temporal_dynamic_relation_emb = dynamic_relation_emb.temporal[:, -1, :, :].to(args.device)  # [:, -1, :, :] to retrieve last hidden from rnn

        # Compute the inter-event time loss for the subgraph.
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


# Compute Loss for static model
def compute_loss_static(model, batch_G, static_entity_emb, args, batch_eid=None):
    """
    Compute the edge loss for the given batch of data in the static RGCN model.

    Args:
        model (StaticRGCNModel): The DGL model to use for computing the loss.
        batch_G (DGLGraph): The batch of data to compute the loss for.
        static_entity_emb (StaticEntityEmb): The static entity embeddings.
        args (argparse.Namespace): The command-line arguments.
        batch_eid (torch.Tensor, optional): The batch of edge IDs to compute the loss for. Defaults to None.

    Returns:
        dict: A dictionary containing the computed edge loss.
    """

    # If batch_eid is not None, create a subgraph of batch_G using only the specified edges.
    if batch_eid is not None:
        assert len(batch_eid) > 0, batch_eid.shape
        sub_batch_G = dgl.edge_subgraph(batch_G, batch_eid.type(settings.DGL_GRAPH_ID_TYPE), preserve_nodes=False)
        sub_batch_G.ndata[dgl.NID] = batch_G.ndata[dgl.NID][sub_batch_G.ndata[dgl.NID].long()]  # map nid in sub_batch_G to nid in the full graph
        sub_batch_G = sub_batch_G.to(args.device)
        batch_eid = None  # Set batch_eid to None to prevent further edge selection in the loss functions below.
    else:
        sub_batch_G = batch_G.to(args.device)

    # Set the number of relations and all nodes in the subgraph.
    sub_batch_G.num_relations = batch_G.num_relations
    sub_batch_G.num_all_nodes = batch_G.num_all_nodes

    # Initialize the loss dictionary.
    loss_dict = {}

    # Get the static entity embeddings for the subgraph.
    sub_batch_G_static_entity_emb = static_entity_emb[sub_batch_G.ndata[dgl.NID].long()].to(args.device)

    # Compute the edge loss for the subgraph.
    edge_LL = model.edge_model(sub_batch_G, sub_batch_G_static_entity_emb, eid=batch_eid)
    loss_dict['edge'] = -edge_LL

    return loss_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, type=str, choices=settings.ALL_GRAPHS, help='Name of graph/dataset')
    parser.add_argument('--seed', type=int, default=101, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use. Set to -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.00001, help="weight decay")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--early-stop', action='store_true', default=True, help="indicates whether to apply early stoping or not")
    parser.add_argument('--patience', type=int, default=30, help="number of times to be tolerate until early stopping")
    parser.add_argument('--early-stop-criterion', type=str, default='MRR', choices=['MRR', 'MAE', 'loss'])
    parser.add_argument('--eval-every', type=int, default=1, help="perform evaluation every k epoch(s)")
    parser.add_argument('--eval-from', type=int, default=0, help="perform evaluation starting from the given epoch")
    parser.add_argument('--full-link-pred-validation', dest='full_link_pred_validation', action='store_true', help="if given, perform full link prediction for validation")
    parser.set_defaults(full_link_pred_validation=False)
    parser.add_argument('--full-link-pred-test', dest='full_link_pred_test', action='store_true', help="if given, perform full link prediction for testing")
    parser.set_defaults(full_link_pred_test=False)
    parser.add_argument('--time-pred-eval', dest='time_pred_eval', action='store_true', help="if given, perform time prediction")
    parser.set_defaults(time_pred_eval=False)
    parser.add_argument('--eval', choices=['edge', 'time', 'both'], default='edge')
    parser.add_argument('--optimize', choices=['edge', 'time', 'both'], default='edge')
    parser.add_argument('--load-best', default=None, help="'edge', 'time', 'both', or path of the best model")
    parser.add_argument('--best-checkpoint-prefix', default=None, type=str, help="prefix of the checkpoint file with the overall best model")
    parser.add_argument('--run-best-checkpoint-prefix', default=None, type=str, help="prefix of the checkpoint file with the run best model")
    parser.add_argument('--clean-up-run-best-checkpoint', dest='clean_up_run_best_checkpoint', action='store_true',
                        help="if given, remove the run best checkpoint at the end of the run")
    parser.set_defaults(clean_up_run_best_checkpoint=False)
    parser.add_argument('--result-file-prefix', type=str, default="")
    parser.add_argument('--log-dir', type=str, default='evokg', help="log directory name")
    parser.add_argument('--embedding-updater-structural-gconv', type=nullable_string,
                        choices=['RGCN+RNN', 'RGCN+GRU', None], default='RGCN+RNN', help="graph conv module for embedding updater")
    parser.add_argument('--embedding-updater-temporal-gconv', type=nullable_string,
                        choices=['RGCN+RNN', 'RGCN+GRU', None], default='RGCN+RNN', help="graph conv module for embedding updater")
    parser.add_argument('--embedding-updater-activation', type=activation_string, default='tanh', help="activation for embedding updater")
    parser.add_argument('--num-gconv-layers', type=int, default=2, help='number of graph convolution layers')
    parser.add_argument('--num-rnn-layers', type=int, default=1, help='number of recurrent layers')
    parser.add_argument('--rnn-truncate-every', type=int, default=20)
    parser.add_argument('--combiner-gconv', type=nullable_string, default=None, help="graph conv module for combiner")
    parser.add_argument('--combiner-activation', type=activation_string, default='tanh', help="activation for combiner")
    parser.add_argument('--static-dynamic-combine-mode', choices=['concat'], default='concat')
    parser.add_argument('--inter-event-time-mode', choices=['node2node_inter_event_times', 'min_inter_event_times'],
                        default='node2node_inter_event_times')
    parser.add_argument('--time-interval-no-log-transform', dest='time_interval_log_transform', action='store_false')
    parser.set_defaults(time_interval_log_transform=True)
    parser.add_argument('--num-mix-components', type=int, default=128)
    parser.add_argument('--static-entity-embed-dim', type=int, default=200)
    parser.add_argument('--structural-dynamic-entity-embed-dim', type=int, default=200)
    parser.add_argument('--temporal-dynamic-entity-embed-dim', type=int, default=200)
    parser.add_argument('--rel-embed-dim', type=int, default=200)

    args = parser.parse_args()
    setup_cuda(args)
    args.exp_time = datetime.now().strftime("%y%m%d_%H%M%S")
    add_logger_file_handler(args.graph, "DKG", args.log_dir, args.exp_time, fname_prefix=args.result_file_prefix)

    main(args)
