# Adapted, with modifications, from https://github.com/DequanWang/tent (MIT License, Copyright (c) 2021 Dequan Wang and Evan Shelhamer).
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.entropy import _softmax_entropy
from src.utils.param_utils import _collect_ln_params, _configure_norm_only


class Tent(nn.Module):
    """Tent: entropy minimisation on all norm-layer affine params (Wang et al., ICLR 2021).
    Official repo: https://github.com/DequanWang/tent
    """

    def __init__(self, model: nn.Module, steps: int = 1, lr: float = 1e-3, momentum: float = 0.9):
        super().__init__()
        self.model = model
        self.steps = max(1, int(steps))
        params = _configure_norm_only(model, _collect_ln_params)
        self.optimizer = (
            torch.optim.SGD(params, lr=lr, momentum=momentum) if params else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.model(x)
        logits = None
        for _ in range(self.steps):
            with torch.enable_grad():
                out = self.model(x)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                self.optimizer.zero_grad(set_to_none=True)
                _softmax_entropy(logits).mean().backward()
                self.optimizer.step()
        return logits
