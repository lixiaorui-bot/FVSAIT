# Lookahead optimizer (Zhang et al., "Lookahead Optimizer: k steps forward,
# 1 step back", NeurIPS 2019), as used by FVSAIT (k=5, gamma=0.5).
#
# The wrapper holds two sets of weights: the fast weights (updated by the
# inner optimizer, e.g. Adam) and the slow weights. Every k fast updates the
# slow weights are interpolated towards the fast ones (Eq. 4 of the FVSAIT
# paper: phi = phi + gamma * (theta - phi)) and the fast weights are reset to
# the slow weights. The slow weights are used as the final model weights.

from collections import defaultdict

import torch


class Lookahead(torch.optim.Optimizer):
    """Lookahead wrapper around any inner optimizer.

    Args:
        optimizer: inner (fast) optimizer, e.g. torch.optim.Adam
        k: number of fast-weight updates before each slow-weight update
        alpha (gamma): interpolation factor for the slow-weight update
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.defaults = dict(optimizer.defaults)
        self._step_count = 0

        # slow weights (one copy per param group)
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]]
            for group in self.param_groups
        ]
        for group_slow in self.slow_weights:
            for q in group_slow:
                q.requires_grad = False

    # -- core -------------------------------------------------------------
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group, group_slow in zip(self.param_groups, self.slow_weights):
                for p, q in zip(group["params"], group_slow):
                    if p.grad is None:
                        continue
                    # phi = phi + alpha * (theta - phi)
                    q.data.add_(p.data.detach() - q.data, alpha=self.alpha)
                    p.data.copy_(q.data)
        return loss

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # -- (de)serialization for checkpointing ------------------------------
    def state_dict(self):
        return {
            "inner": self.optimizer.state_dict(),
            "slow": [
                [q.data for q in group_slow] for group_slow in self.slow_weights
            ],
            "k": self.k,
            "alpha": self.alpha,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["inner"])
        for group_slow, saved in zip(self.slow_weights, state_dict["slow"]):
            for q, s in zip(group_slow, saved):
                q.data.copy_(s)
        self.k = state_dict.get("k", self.k)
        self.alpha = state_dict.get("alpha", self.alpha)
        self._step_count = state_dict.get("step_count", 0)

    # -- passthrough helpers ----------------------------------------------
    def __repr__(self):
        return f"Lookahead({self.optimizer!r}, k={self.k}, alpha={self.alpha})"

    def add_param_group(self, param_group):
        param_group = self.optimizer.add_param_group(param_group)
        self.slow_weights.append(
            [p.clone().detach() for p in param_group["params"]]
        )
        return param_group
