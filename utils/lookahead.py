# Lookahead optimizer wrapper
# Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
# 每 k 步内层(inner)优化器更新后, 将快权重向慢权重插值: slow <- slow + alpha * (fast - slow),
# 然后将快权重回退到慢权重.

from collections import defaultdict

import torch


class Lookahead(torch.optim.Optimizer):
    """
    Wrap an inner optimizer with the Lookahead mechanism.

    Args:
        optimizer: inner optimizer (e.g., torch.optim.Adam)
        k: number of inner (fast) steps between slow-weight updates
        alpha: slow-weight interpolation step size
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        if k < 1:
            raise ValueError("Lookahead k must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Lookahead alpha must be in [0, 1]")
        self.optimizer = optimizer
        self.k = int(k)
        self.alpha = float(alpha)
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = defaultdict(dict)
        self._step_count = 0

    @property
    def _slow_params(self):
        # 按参数顺序列出慢权重(state_dict 用)
        out = []
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                out.append(st.get("slow_param"))
        return out

    def _update_slow(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                slow = st.get("slow_param")
                if slow is None:
                    st["slow_param"] = p.detach().clone()
                else:
                    slow.add_(p.detach() - slow, alpha=self.alpha)
                p.data.copy_(st["slow_param"])

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            self._update_slow()
        return loss

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "inner": self.optimizer.state_dict(),
            "slow_params": [None if s is None else s.detach().cpu() for s in self._slow_params],
            "k": self.k,
            "alpha": self.alpha,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["inner"])
        self.k = state_dict.get("k", self.k)
        self.alpha = state_dict.get("alpha", self.alpha)
        self._step_count = state_dict.get("step_count", 0)
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                t = state_dict["slow_params"][i]
                i += 1
                if t is not None:
                    self.state[p] = {"slow_param": t.to(device=p.device, dtype=p.dtype)}
