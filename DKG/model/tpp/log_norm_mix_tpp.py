# Adapted and revised code from https://github.com/shchur/ifl-tpp/blob/master/code/dpp/models/log_norm_mix.py
import dgl
import torch
import torch.distributions as D
import torch.nn as nn

from DKG.model.time_interval_transform import TimeIntervalTransform
from torch.distributions import Normal, MixtureSameFamily, TransformedDistribution


class LogNormMixTPP(nn.Module):
    """
    TPP model where the distribution of inter-event times given the history is modeled using a LogNormal mixture distribution
    """

    def __init__(
        self,
        dynamic_entity_embed_dim,
        static_entity_embed_dim,
        num_rels,
        rel_embed_dim,
        mode,
        num_mix_components,
        time_interval_transform,
        mean_log_inter_event_time=0.0,
        std_log_inter_event_time=1.0,
        dropout=0.0,
    ):
        super().__init__()

        self.dynamic_entity_embed_dim = dynamic_entity_embed_dim
        self.static_entity_embed_dim = static_entity_embed_dim
        self.num_rels = num_rels
        self.rel_embed_dim = rel_embed_dim
        self.rel_embeds = nn.Parameter(torch.Tensor(self.num_rels, rel_embed_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        self.mode = mode
        self.num_mix_components = num_mix_components
        self.time_interval_transform = time_interval_transform
        assert isinstance(self.time_interval_transform, TimeIntervalTransform)
        self.mean_log_inter_event_time = mean_log_inter_event_time
        self.std_log_inter_event_time = std_log_inter_event_time
        self.dropout = nn.Dropout(dropout)

        self.context_size = LogNormMixTPP.get_context_size(dynamic_entity_embed_dim, static_entity_embed_dim,
                                                           rel_embed_dim, mode)
        self.context_to_params = nn.Sequential(
            nn.Linear(self.context_size, self.context_size),
            nn.Tanh(),
            nn.Linear(self.context_size, 3 * self.num_mix_components)
        )

    def get_inter_event_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.
        """
        context = self.dropout(context)
        raw_params = self.context_to_params(context)
        # slice the `raw_params` tensor to get the parameters of the mixture distribution
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_weights = torch.log_softmax(log_weights, dim=-1)
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_event_time=self.mean_log_inter_event_time,
            std_log_inter_event_time=self.std_log_inter_event_time,
        )

    def get_inter_event_time(self, batch_G, node_latest_event_time, batch_eid=None):
        global_inter_event_time = True if self.mode == 'min_inter_event_times' else False
        from DKG.model.embedding import EventTimeHelper

        recipient_inter_event_time = EventTimeHelper.get_sparse_inter_event_times(  # shape: (# edges in batch_G,)
            batch_G, node_latest_event_time[..., 0], _global=global_inter_event_time
        )
        rev_batch_G = dgl.reverse(batch_G, copy_ndata=True, copy_edata=True)
        rev_batch_G.num_relations = batch_G.num_relations
        rev_batch_G.num_all_nodes = batch_G.num_all_nodes
        sender_inter_event_time = EventTimeHelper.get_sparse_inter_event_times(  # shape: (# edges in batch_G,)
            rev_batch_G, node_latest_event_time[..., 1], _global=global_inter_event_time
        )
        assert len(recipient_inter_event_time) == len(sender_inter_event_time) == batch_G.num_edges(), (len(recipient_inter_event_time), batch_G.num_edges())

        if batch_eid is not None:
            recipient_inter_event_time = recipient_inter_event_time[batch_eid]
            sender_inter_event_time = sender_inter_event_time[batch_eid]

        if self.mode == 'node2node_inter_event_times':
            sender_inter_event_time_ = self.time_interval_transform(sender_inter_event_time)
            recipient_inter_event_time_ = self.time_interval_transform(recipient_inter_event_time)
            assert torch.all(sender_inter_event_time_.eq(recipient_inter_event_time_)), \
                (sender_inter_event_time, sender_inter_event_time_, recipient_inter_event_time, recipient_inter_event_time_)

            return sender_inter_event_time_.view(-1, 1).clamp(min=1e-10)
        elif self.mode == 'min_inter_event_times':
            sender_inter_event_time_ = self.time_interval_transform(sender_inter_event_time)
            recipient_inter_event_time_ = self.time_interval_transform(recipient_inter_event_time)

            m = torch.max(sender_inter_event_time_, recipient_inter_event_time_)
            return m.view(-1, 1).clamp(min=1e-10)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def get_context(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb, batch_eid=None):
        src, dst = batch_G.edges()
        rel = batch_G.edata['rel_type']
        if batch_eid is not None:
            src, dst = src[batch_eid], dst[batch_eid]
            rel = rel[batch_eid]
        rel = rel.long()

        recipient_dynamic_entity_emb = dynamic_entity_emb[..., 0]
        sender_dynamic_entity_emb = dynamic_entity_emb[..., 1]
        # noinspection PyArgumentList
        dyn_rel_emb = torch.mean(torch.stack([dynamic_relation_emb[:, :, 1], dynamic_relation_emb[:, :, 0]]), axis=0)

        if self.mode in ['node2node_inter_event_times', 'min_inter_event_times']:
            return torch.cat((sender_dynamic_entity_emb[src.long()], recipient_dynamic_entity_emb[dst.long()],
                              self.rel_embeds[rel], dyn_rel_emb[rel]), dim=1).unsqueeze(1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @classmethod
    def get_context_size(cls, dynamic_entity_embed_dim, static_entity_embed_dim, rel_embed_dim, mode):
        if mode in ['node2node_inter_event_times', 'min_inter_event_times']:
            return 2 * dynamic_entity_embed_dim + 2 * rel_embed_dim
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def expected_event_time(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                            node_latest_event_time, batch_eid=None):
        ctx = self.get_context(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb, batch_eid)
        inter_event_time_dist = self.get_inter_event_time_dist(ctx)
        expected_inter_event_time = inter_event_time_dist.mean.view(-1)  # shape=(# edges in batch_G,)
        expected_inter_event_time = self.time_interval_transform.reverse_transform(expected_inter_event_time)

        from DKG.model.embedding import EventTimeHelper
        global_inter_event_time = True if self.mode == 'min_inter_event_times' else False
        latest_event_time = EventTimeHelper.get_sparse_latest_event_times(  # shape=(# edges in batch_G,)
            batch_G, node_latest_event_time[..., 0], _global=global_inter_event_time
        ).view(-1)

        return latest_event_time.cpu() + expected_inter_event_time.cpu()

    def log_prob_interval(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                          node_latest_event_time, batch_eid=None, reduction=None):
        assert batch_G.num_nodes() == len(dynamic_entity_emb) == len(static_entity_emb), (batch_G.num_nodes(), len(dynamic_entity_emb), len(static_entity_emb))

        if self.mode in ['node2node_inter_event_times', 'min_inter_event_times']:
            ctx = self.get_context(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb, batch_eid)
            inter_event_time_dist = self.get_inter_event_time_dist(ctx)
            inter_event_time = self.get_inter_event_time(batch_G, node_latest_event_time, batch_eid)
            assert torch.all(inter_event_time >= 0)

            if self.time_interval_transform.log_transform:
                iet0 = (torch.exp(inter_event_time) - batch_G.time_interval).clamp(1e-20)
                iet0 = torch.log(iet0).clamp(1e-20)
                prob_interval = inter_event_time_dist.cdf(inter_event_time) - inter_event_time_dist.cdf(iet0)
            else:
                iet0 = (inter_event_time - batch_G.time_interval).clamp(1e-20)
                prob_interval = inter_event_time_dist.cdf(inter_event_time) - inter_event_time_dist.cdf(iet0)
            assert torch.all(prob_interval.view(-1) >= 0)

            if reduction is None:
                return torch.log(prob_interval.clamp(min=1e-20))
            else:
                raise ValueError(f"Invalid reduction: {reduction}")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def log_prob_density(self, batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb,
                         node_latest_event_time, batch_eid=None, reduction=None):
        if self.mode in ['node2node_inter_event_times', 'min_inter_event_times']:
            assert batch_G.num_nodes() == len(dynamic_entity_emb) == len(static_entity_emb), (batch_G.num_nodes(), len(dynamic_entity_emb), len(static_entity_emb))
            ctx = self.get_context(batch_G, dynamic_entity_emb, static_entity_emb, dynamic_relation_emb, batch_eid)
            inter_event_time_dist = self.get_inter_event_time_dist(ctx)
            inter_event_time = self.get_inter_event_time(batch_G, node_latest_event_time, batch_eid)
            log_p = inter_event_time_dist.log_prob(inter_event_time)

            if reduction is None:
                return log_p
            elif reduction == 'mean':
                return log_p.mean()
            else:
                raise ValueError(f"Invalid reduction: {reduction}")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def extra_repr(self):
        field_desc = [f"mode={self.mode}", f"context_size={self.context_size}",
                      f"rel_embed_dim={self.rel_embed_dim}", f"num_mix_components={self.num_mix_components}"]
        return ", ".join(field_desc)


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions, which is modeled as follows.

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_event_time * x + mean_log_inter_event_time
    z = exp(y)
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_event_time: float = 0.0,
        std_log_inter_event_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_event_time == 0.0 and std_log_inter_event_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_event_time, scale=std_log_inter_event_time)]
        self.mean_log_inter_event_time = mean_log_inter_event_time
        self.std_log_inter_event_time = std_log_inter_event_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.
        """
        a = self.std_log_inter_event_time
        b = self.mean_log_inter_event_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()
