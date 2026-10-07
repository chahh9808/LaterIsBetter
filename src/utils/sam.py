from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer (Foret et al., 2021)."""

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p].pop("old_p")
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        grads = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grads:
            return torch.tensor(0.0)
        device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack(grads).to(device), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
