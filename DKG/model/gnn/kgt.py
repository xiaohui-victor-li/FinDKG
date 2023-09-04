"""
This module implements the Knowledge Graph Transformer and Graph Transformer classes for use in dynamic knowledge graph embedding. 

The Knowledge Graph Transformer is a multi-layer transformer that takes in a graph, node embeddings, node types, and edge types, and outputs updated node embeddings. The Graph Transformer is a single-layer transformer that is used as a building block for the Knowledge Graph Transformer.

This module requires the following packages: torch, torch.nn, dgl, dgl.function, dgl.nn.pytorch.linear, dgl.nn.pytorch.softmax, and math.
"""
# Knowledge Graph Transformer
# ---------------------------
#
# add SwiGLU activation
#

import torch
import torch.nn as nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear
from dgl.nn.pytorch.softmax import edge_softmax

import math


class KGTransformer(nn.Module):
    """ Knowledge Graph Transformer
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_heads,
                 out_dim,
                 n_layers,
                 num_nodes,
                 num_rels,
                 dropout=0.2,
                 layer_norm=True,
                 low_mem=False):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads  # number of attention heads
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.activation = torch.nn.SiLU()
        self.feed_forward = nn.Linear(self.hid_dim, self.out_dim)
        self.low_mem = low_mem

        # stacke Transformer layers
        self.n_layers = n_layers
        assert self.n_layers >= 1, self.n_layers
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(GraphTransformer(
                self.in_dim, self.hid_dim, self.num_heads,
                num_ntypes = self.num_nodes,
                num_etypes = self.num_rels,
                dropout=self.dropout,
                use_norm=self.layer_norm#, low_mem=low_mem,
            ))
        else:
            # i2h
            self.layers.append(GraphTransformer(
                self.in_dim, self.hid_dim, self.num_heads,
                num_ntypes = self.num_nodes,
                num_etypes = self.num_rels,
                dropout=self.dropout,
                use_norm=self.layer_norm#, low_mem=low_mem,
            ))
            # h2h
            for i in range(1, self.n_layers):
                self.layers.append(GraphTransformer(
                    self.hid_dim, self.hid_dim, self.num_heads,
                    num_ntypes = self.num_nodes,
                    num_etypes = self.num_rels,
                    dropout=self.dropout,
                    use_norm=self.layer_norm#, low_mem=low_mem,
                ))
        assert self.n_layers == len(self.layers), (self.n_layers, len(self.layers))

    def forward(self, G, emb, ntypes, etypes, norm=None, use_norm=True):
        """Forward computation.
        """
        for layer in self.layers:
            emb = layer(G, emb, ntypes, etypes, norm=norm)
        output_emb = self.feed_forward(emb)
        return output_emb 





class GraphTransformer(nn.Module):
    r"""Knowledge graph transformer Layer

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node-level features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
      ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
        \frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
      K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
      Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
      \cdot Message(s,e,t)\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """

    def __init__(
        self,
        in_size,
        hid_size,
        num_heads,
        num_ntypes,
        num_etypes,
        dropout=0.2,
        use_norm=True,
    ):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.num_heads = num_heads
        head_size = self.hid_size//self.num_heads
        self.head_size = head_size
        self.sqrt_d = math.sqrt(self.head_size)
        self.use_norm = use_norm

        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)

        # linear projection A-Linear
        self.activation = torch.nn.SiLU()
        self.linear_a = TypedLinear(
            head_size * num_heads, head_size * num_heads, num_ntypes
        )

        self.relation_pri = nn.ParameterList(
            [nn.Parameter(torch.ones(num_etypes)) for i in range(num_heads)]
        )
        self.relation_att = nn.ModuleList(
            [
                TypedLinear(head_size, head_size, num_etypes)
                for i in range(num_heads)
            ]
        )
        self.relation_msg = nn.ModuleList(
            [
                TypedLinear(head_size, head_size, num_etypes)
                for i in range(num_heads)
            ]
        )
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(
                torch.Tensor(in_size, head_size * num_heads)
            )
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[: g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[: g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            q = self.linear_q(x_dst, dstntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            v = self.linear_v(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            g.srcdata["k"] = k
            g.dstdata["q"] = q
            g.srcdata["v"] = v
            g.edata["etype"] = etype
            g.apply_edges(self.message)
            g.edata["m"] = g.edata["m"] * edge_softmax(
                g, g.edata["a"]
            ).unsqueeze(-1)

            # appen norm
            if norm is not None:
                g.edata['norm'] = norm

            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)
            # target-specific aggregation
            h = self.drop(self.activation(self.linear_a(h, dstntype, presorted)))

            ## Residual Connection
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)

            return h


    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data["etype"]
        k = torch.unbind(edges.src["k"], dim=1)
        q = torch.unbind(edges.dst["q"], dim=1)
        v = torch.unbind(edges.src["v"], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            a.append(
                (kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d
            )  # (E,)

            # message at head
            m_head = self.relation_msg[i](v[i], etype, self.presorted)

            ## * message-passing edge normalization
            if 'norm' in edges.data:
                m_head = m_head * edges.data['norm']
            # (E, O)
            m.append(
                m_head
            )  
        return {"a": torch.stack(a, dim=1), "m": torch.stack(m, dim=1)}
