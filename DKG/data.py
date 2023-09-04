# Data module for loading knowledge graphs
# ----------------------------------------
#
import os

import dgl
import numpy as np
import pandas as pd
import torch

# Graph Configuraton
from DKG import settings


def load_temporal_knowledge_graph(graph_name, idtype=settings.DGL_GRAPH_ID_TYPE, data_root=settings.DATA_ROOT):
    if graph_name in settings.ALL_GRAPHS:
        train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    else:
        raise ValueError(f"Invalid graph name: {graph_name}")

    column_names = ['head', 'rel', 'tail', 'time', '_']
    train_data_table = load_data_table(graph_name, train_file, column_names, data_root=data_root)
    val_data_table = load_data_table(graph_name, val_file, column_names, data_root=data_root)
    test_data_table = load_data_table(graph_name, test_file, column_names, data_root=data_root)
    all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)

    stat_table = load_data_table(graph_name, "stat.txt",
                                 column_names=['num_entities', 'num_relations', '_'],
                                 data_root=data_root)
    num_entities, num_relations = stat_table['num_entities'].item(), stat_table['num_relations'].item()

    heads = torch.from_numpy(all_data_table['head'].to_numpy())
    tails = torch.from_numpy(all_data_table['tail'].to_numpy())
    rels = torch.from_numpy(all_data_table['rel'].to_numpy())
    times = torch.from_numpy(all_data_table['time'].to_numpy())

    # create a [dgl] graph and add edge masks
    G = dgl.graph((heads, tails), num_nodes=num_entities, idtype=idtype)
    G.name = graph_name
    G.num_relations = num_relations
    # noinspection PyTypeChecker
    G.edata['rel_type'] = rels.type(idtype)

    # add Node type data as knowledge entity category
    if graph_name in ['FinDKG', 'FinDKG-live']:
        G.num_node_types = 12
        print(" --> Load the entity type data for FinDKG")
        node_data_table = load_data_table(graph_name, "entity2id.txt",
                                          column_names=['entity', 'entity_id', 'ntype', 'ntype_id'],
                                          data_root=data_root)
        G.ndata['node_type'] =  torch.from_numpy(node_data_table['ntype_id'].to_numpy()).type(idtype)
    else:
        G.num_node_types = 1
        G.ndata['node_type'] = torch.zeros(num_entities).type(idtype)

    G.edata['time'] = times.float()
    time_diff = np.diff(torch.unique(G.edata['time']).tolist())
    G.time_interval = min(time_diff)

    G.train_times = np.sort(np.unique(train_data_table['time'].to_numpy()))
    G.val_times = np.sort(np.unique(val_data_table['time'].to_numpy()))
    G.test_times = np.sort(np.unique(test_data_table['time'].to_numpy()))

    num_edges = len(all_data_table)
    train_edge_mask = get_edge_mask(num_edges, 0, len(train_data_table))
    val_edge_mask = get_edge_mask(num_edges, len(train_data_table), len(train_data_table) + len(val_data_table))
    test_edge_mask = get_edge_mask(num_edges, len(train_data_table) + len(val_data_table), num_edges)

    G.edata['train_mask'] = train_edge_mask
    G.edata['val_mask'] = val_edge_mask
    G.edata['test_mask'] = test_edge_mask

    return G


def load_data_table(graph_name, file_name, column_names=None, data_root=settings.DATA_ROOT):
    data_fpath = os.path.join(data_root, graph_name, file_name)
    return pd.read_table(data_fpath, sep='\t', names=column_names)


def get_edge_mask(num_edges, edge_index_from, edge_index_until):
    """
    return binary edge masks for edges from edge_index_from (inclusive) till edge_index_until (exclusive)
    """
    assert 0 <= edge_index_from < edge_index_until <= num_edges
    mask = torch.zeros(num_edges, dtype=torch.bool)
    mask[edge_index_from: edge_index_until] = True
    return mask

if __name__ == '__main__':
    G = load_temporal_knowledge_graph(settings.GRAPH_ICEWS18)
