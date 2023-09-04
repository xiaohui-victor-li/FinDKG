import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv



class RGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 n_layers,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 use_bias=True,
                 activation=F.relu,
                 use_self_loop=True,
                 dropout=0.0,
                 layer_norm=False,
                 low_mem=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.use_bias = use_bias
        self.activation = activation
        self.use_self_loop = use_self_loop
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.low_mem = low_mem

        self.n_layers = n_layers
        assert self.n_layers >= 1, self.n_layers
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=None, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm#, low_mem=low_mem,
            ))
        else:
            # i2h
            self.layers.append(RelGraphConv(
                self.in_dim, self.hid_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=self.activation, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm#, low_mem=low_mem,
            ))
            # h2h
            for i in range(1, self.n_layers - 1):
                self.layers.append(RelGraphConv(
                    self.hid_dim, self.hid_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                    activation=self.activation, self_loop=self.use_self_loop, dropout=self.dropout,
                    layer_norm=self.layer_norm#, low_mem=low_mem,
                ))
            # h2o
            self.layers.append(RelGraphConv(
                self.hid_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=None, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm#, low_mem=low_mem,
            ))
        assert self.n_layers == len(self.layers), (self.n_layers, len(self.layers))

    def forward(self, G, emb, etypes, edge_norm=None):
        if edge_norm is not None:
            edge_norm = edge_norm.view(-1, 1)

        for layer in self.layers:
            emb = layer(G, emb, etypes, edge_norm)
        return emb
