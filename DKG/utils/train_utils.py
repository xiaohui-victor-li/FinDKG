import os

import torch
from torch import nn as nn

from DKG import settings
from DKG.utils.log_utils import logger


class EarlyStopping:
    def __init__(self,
                 network,
                 patience=30,
                 result_root=settings.RESULT_ROOT,
                 run_best_checkpoint_prefix="run_best_checkpoint",
                 overall_best_checkpoint_prefix="overall_best_checkpoint",
                 eval="edge",
                 minimizing_objective=False,
                 logging=True):
        self.network = network
        self.patience = patience
        self.run_best_checkpoint_prefix = run_best_checkpoint_prefix
        self.overall_best_checkpoint_prefix = overall_best_checkpoint_prefix
        self.eval = eval
        self.result_root = result_root
        self.minimizing_objective = minimizing_objective
        self.counter = 0
        self.early_stop = False
        self.logging = logging
        self.run_best_score = None
        self.overall_best_score = self.load_overall_best_score()

    @property
    def run_best_checkpoint_fpath(self, eval=None):
        if eval is None:
            eval = self.eval
        return os.path.join(self.result_root, f"{self.run_best_checkpoint_prefix}_opt_{eval}.pt")

    def overall_best_checkpoint_fpath(self, eval=None):
        if eval is None:
            eval = self.eval
        return os.path.join(self.result_root, f"{self.overall_best_checkpoint_prefix}_opt_{eval}.pt")

    def load_overall_best_score(self):
        try:
            overall_best_score = self.load_checkpoint(self.overall_best_checkpoint_fpath())['score']
        except Exception:
            overall_best_score = None
        if self.logging:
            logger.info("=" * 100)
            logger.info(f"Overall best score ({self.overall_best_checkpoint_fpath()}) = {overall_best_score}")
            logger.info("=" * 100)
        return overall_best_score

    def step(self, score, model_state=None):
        """Return whether to early stop"""

        if self.run_best_score is None or self.improved(score, self.run_best_score):
            self.run_best_score = score
            if self.logging:
                logger.info(f"[EarlyStopping] Best validation score updated to {self.run_best_score:.4f}")
            if model_state is not None:
                model_state['score'] = self.run_best_score
                self.save_checkpoint(model_state, self.run_best_checkpoint_fpath)
            self.counter = 0
        else:
            self.counter += 1
            if self.logging:
                logger.info(f"[EarlyStopping] counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        if self.overall_best_score is None or self.improved(score, self.overall_best_score):
            self.overall_best_score = score
            if self.logging:
                logger.info(f"Overall best validation score updated to {self.overall_best_score:.4f}")
            if model_state is not None:
                model_state['score'] = self.overall_best_score
                self.save_checkpoint(model_state, self.overall_best_checkpoint_fpath())

        return self.early_stop

    def improved(self, score, best_score):
        if self.minimizing_objective:
            return True if score < best_score else False
        else:
            return True if score > best_score else False

    def save_checkpoint(self, model_state, checkpoint_fpath=None):
        if checkpoint_fpath is None:
            checkpoint_fpath = self.run_best_checkpoint_fpath
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)

        torch.save(model_state, checkpoint_fpath)

    def load_checkpoint(self, checkpoint_fpath=None):
        if checkpoint_fpath is None:
            checkpoint_fpath = self.run_best_checkpoint_fpath
        return torch.load(checkpoint_fpath)


def nullable_string(val):
    if not val or val.lower() in ['none', 'null']:
        return None
    return val


def activation_string(val):
    val = nullable_string(val)
    if val is None:
        return val
    activation_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
    return activation_dict[val]


def setup_cuda(args):
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()
