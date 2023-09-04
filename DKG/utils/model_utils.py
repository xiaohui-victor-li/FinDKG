import dgl
import torch
from torch import nn as nn


# noinspection PyTypeChecker
def collate_fn(batch_times, G):
    time_from, time_to = batch_times[0], batch_times[-1]
    edges_before_batch = torch.nonzero(G.edata['time'] < time_from, as_tuple=False).squeeze().type(G.idtype)
    edges_in_batch = torch.nonzero((time_from <= G.edata['time']) & (G.edata['time'] <= time_to), as_tuple=False).squeeze().type(G.idtype)
    edges_until_batch = torch.nonzero(G.edata['time'] <= time_to, as_tuple=False).squeeze().type(G.idtype)

    def copy_attr(target_G):
        target_G.num_relations = G.num_relations
        target_G.num_all_nodes = G.num_nodes()
        target_G.time_interval = G.time_interval

    prior_G = dgl.edge_subgraph(G, edges_before_batch) #  preserve_nodes=False
    prior_G.ndata['norm'] = comp_deg_norm(prior_G)  # used by R-GCN
    copy_attr(prior_G)

    batch_G = dgl.edge_subgraph(G, edges_in_batch)   # , preserve_nodes=False
    batch_G.ndata['norm'] = comp_deg_norm(batch_G)  # used by R-GCN
    copy_attr(batch_G)

    cumul_G = dgl.edge_subgraph(G, edges_until_batch)  # , preserve_nodes=False
    cumul_G.ndata['norm'] = comp_deg_norm(cumul_G)  # used by R-GCN
    copy_attr(cumul_G)

    return prior_G, batch_G, cumul_G, batch_times


def comp_deg_norm(G):
    in_deg = G.in_degrees(range(G.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg

    return norm


def node_norm_to_edge_norm(G):
    G = G.local_var()
    G.apply_edges(lambda edges: {'norm': edges.dst['norm']})

    return G.edata['norm']


def get_embedding(num_embeddings, embedding_dims, zero_init=False, device=None):
    if type(embedding_dims) == int:
        embedding_dims = [embedding_dims]

    if zero_init:
        embed = nn.Parameter(torch.zeros(num_embeddings, *embedding_dims))
    else:
        embed = nn.Parameter(torch.Tensor(num_embeddings, *embedding_dims))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))

    if device is not None:
        embed = embed.to(device)

    return embed
