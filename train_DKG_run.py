# Training scrip for KGTransformer model
# --------------------------------------
#
# Train cml:
#   python train_DKG_run.py 
#

import os
import sys
import pprint

import pandas as pd
import numpy as np

import dgl
import time

from tqdm import tqdm
from functools import partial
from copy import deepcopy
from collections import defaultdict
from datetime import datetime


### Import the DKG library
# Get the absolute path of the current script
current_path = os.path.abspath(__file__)
print("Current path:", current_path)
# Move up two levels
parent_parent_path = os.path.dirname(os.path.dirname(current_path))
print("Two levels up:", parent_parent_path)
sys.path.append(parent_parent_path)


#################### DKG Library ####################
import torch
from torch.utils.data import DataLoader

import DKG
print(DKG.__version__)

from DKG.utils.log_utils import add_logger_file_handler, get_log_root_path, logger
from DKG.utils.model_utils import get_embedding
from DKG.utils.train_utils import setup_cuda, EarlyStopping, nullable_string, activation_string

from DKG.model import DynamicGraphModel, DKG_DEFAULT_CONFIG, EmbeddingUpdater, Combiner, EdgeModel, InterEventTimeModel, MultiAspectEmbedding
from DKG.model.time_interval_transform import TimeIntervalTransform

from DKG.train import forward_graphs,  pack_checkpoint, unpack_checkpoint, compute_loss
from DKG.eval import evaluate

INTER_EVENT_TIME_DTYPE = torch.float32


############################### Config ############################### 
graph_mode = "FinDKG"   # specify the dataset: "FinDKG" "ICEWS18"  #"ICEWS14"  #"ICEWS_500"  #"GDELT"  #"WIKI"  #"YAGO"

model_ver = "KGTransformer"   # Mode name: "GraphTransformer"
model_type ='KGT+RNN'  # 'KGT+RNN' for GraphTransformer | 'RGCN+RNN' for GraphRNN
epoch_times = 150
random_seed = 41
data_root_path = './data'   # output data path

flag_train = True    # Traing the model
flag_eval = True     # Evaluate the model


############################### Load Graph Data ###############################
G = DKG.data.load_temporal_knowledge_graph(graph_mode, data_root=data_root_path)
collate_fn = partial(DKG.utils.collate_fn, G=G)

train_data_loader = DataLoader(G.train_times, shuffle=False, collate_fn=collate_fn)
val_data_loader = DataLoader(G.val_times, shuffle=False, collate_fn=collate_fn)
test_data_loader = DataLoader(G.test_times, shuffle=False, collate_fn=collate_fn)


############################### Model Config ###############################
args = deepcopy(DKG_DEFAULT_CONFIG)

args.seed = random_seed  # Random Seed 
args.cuda = args.gpu >= 0 and torch.cuda.is_available()
args.device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")  
args.graph = graph_mode
args.version = model_ver #'GTransformer' 
args.optimize = 'both'

dim_num = 200   # default dim set up to 200
args.static_entity_embed_dim = dim_num 
args.structural_dynamic_entity_embed_dim = dim_num 
args.temporal_dynamic_entity_embed_dim = dim_num 

args.num_gconv_layers = 2   # layer of the KGTransformer, default set up 2

args.num_attn_heads = 8
args.lr = 0.0005  # leanrning rate

# Freeze the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device=='cuda':
    torch.cuda.manual_seed_all(args.seed)

num_relations = G.num_relations
    
# Training dir
log_root_path = get_log_root_path(args.graph, args.log_dir)
print(log_root_path)
overall_best_checkpoint_prefix = f"{args.graph}_{args.version}_overall_best_checkpoint"
run_best_checkpoint_prefix = f"{args.graph}_{args.version}_run_best_checkpoint"
    
with open(os.path.join(log_root_path, f"{args.graph}_args_{args.version}.txt"), 'w') as f:
    f.write(pprint.pformat(args.__dict__))

add_logger_file_handler(args.graph, args.version , '.',
                        log_time=None, fname_prefix="")
    
logger.info(f"  --> Set up random seed [ {args.seed} ]")


######################### Build up KG #########################

node_latest_event_time = torch.zeros(G.number_of_nodes(),
                                     G.number_of_nodes() + 1,
                                     2,
                                     dtype=INTER_EVENT_TIME_DTYPE)
time_interval_transform = TimeIntervalTransform(log_transform=True)

# KG embedding module
embedding_updater = EmbeddingUpdater(G.number_of_nodes(),
                                     args.static_entity_embed_dim,
                                     args.structural_dynamic_entity_embed_dim,
                                     args.temporal_dynamic_entity_embed_dim,
                                     node_latest_event_time,
                                     G.num_relations,
                                     args.rel_embed_dim,
                                     num_node_types=G.num_node_types,
                                     num_heads=args.num_attn_heads,   # number of attention head for Transformer
                                     graph_structural_conv=model_type,
                                     graph_temporal_conv=model_type,
                                     num_gconv_layers=args.num_gconv_layers,
                                     num_rnn_layers=args.num_rnn_layers,
                                     time_interval_transform=time_interval_transform,
                                     dropout=args.dropout,
                                     activation=args.embedding_updater_activation,
                                     graph_name=args.graph).to(args.device)

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

model = DynamicGraphModel(embedding_updater, combiner, edge_model, inter_event_time_model, node_latest_event_time).to(args.device)



# Init the static and dynamic entity & relation embeddings
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
####################################################################################################


############################## Graph Model Training ##############################

# > ES lerning system
stopper = EarlyStopping(args.graph, args.patience,
                        result_root=log_root_path,
                        run_best_checkpoint_prefix=run_best_checkpoint_prefix,
                        overall_best_checkpoint_prefix=overall_best_checkpoint_prefix,
                        eval=args.eval)
    
params = list(model.parameters()) + [
        static_entity_embeds.structural, static_entity_embeds.temporal,
        init_dynamic_entity_embeds.structural, init_dynamic_entity_embeds.temporal,
        init_dynamic_relation_embeds.structural, init_dynamic_relation_embeds.temporal,
    ]
edge_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
time_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

######################## > Training secetion ##############################
if flag_train:
    # Prestore the model
    model.eval()
    model.node_latest_event_time.zero_()
    node_latest_event_time.zero_()
    dynamic_entity_emb = init_dynamic_entity_embeds
    dynamic_relation_emb = init_dynamic_relation_embeds

    val_dict, val_dynamic_entity_emb, val_dynamic_relation_emb, loss_weights = \
        evaluate(model, val_data_loader, G, static_entity_embeds, dynamic_entity_emb, dynamic_relation_emb,
                num_relations, args, "Validation", args.full_link_pred_validation, args.time_pred_eval, 0)

    # Save the model locally 
    node_latest_event_time_post_valid = deepcopy(model.node_latest_event_time)
    pack_args = (model, edge_optimizer, time_optimizer, static_entity_embeds,
                            init_dynamic_entity_embeds, val_dynamic_entity_emb,
                            init_dynamic_relation_embeds, val_dynamic_relation_emb,
                            node_latest_event_time_post_valid, loss_weights, args)
    score = val_dict['MRR']
    stopper.step(score, pack_checkpoint(*pack_args))

    # Start training along epochs
    for epoch in range(epoch_times):
        ######### Training ##########
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

        ######### Validation #########
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




#################### > Test Set Evaluation ################################
if flag_eval:
    run_best_checkpoint = stopper.load_checkpoint()

    model, edge_optimizer, time_optimizer, val_static_entity_emb, \
        init_dynamic_entity_embeds, val_dynamic_entity_emb, \
        init_dynamic_relation_embeds, val_dynamic_relation_emb, \
        node_latest_event_time_post_valid, loss_weights, _ = \
            unpack_checkpoint(run_best_checkpoint, model, edge_optimizer, time_optimizer)

    logger.info("Loaded the best model so far for testing.")


    model.node_latest_event_time.copy_(node_latest_event_time_post_valid)

    test_start_time = time.time()
    evaluate(model, test_data_loader, G, val_static_entity_emb, val_dynamic_entity_emb, val_dynamic_relation_emb,
            num_relations, args,
            "Test",   # specify the test set
            full_link_pred_eval=args.full_link_pred_test,
            time_pred_eval=args.time_pred_eval,
            loss_weights=loss_weights)
    test_end_time = time.time()
    logger.info(f"Test elapsed time={test_end_time - test_start_time:.4f} secs")
