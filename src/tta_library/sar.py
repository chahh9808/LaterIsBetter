# Adapted, with modifications, from https://github.com/mr-eggplant/SAR (BSD 3-Clause License, Copyright (c) 2023 Shuaicheng Niu).
from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn

from src.utils.entropy import _softmax_entropy
from src.utils.param_utils import _collect_sar_params, _configure_norm_only
from src.utils.sam import SAM


class SAR(nn.Module):
    """SAR: SAM + reliable-sample entropy filter + EMA model recovery (Niu et al., ICLR 2023).
    Official repo: https://github.com/mr-eggplant/SAR
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
        lr: float = 1e-3,
        momentum: float = 0.9,
        rho: float = 0.05,
        margin_e0: float = 0.4 * math.log(1000),
        reset_constant_em: float = 0.2,
    ):
        super().__init__()
        self.model = model
        self.steps = max(1, int(steps))
        self.margin_e0 = margin_e0
        self.reset_constant_em = reset_constant_em
        self.ema: float | None = None

        params = _configure_norm_only(model, _collect_sar_params)
        self.optimizer = (
            SAM(params, torch.optim.SGD, rho=rho, lr=lr, momentum=momentum)
            if params else None
        )
        self._model_state = deepcopy(model.state_dict())
        self._opt_state = deepcopy(self.optimizer.state_dict()) if self.optimizer else None

    def _reset(self):
        # state dict was saved before model.to(device), so tensors may be on CPU
        device = next(self.model.parameters()).device
        state = {k: v.to(device) for k, v in self._model_state.items()}
        self.model.load_state_dict(state, strict=True)
        if self.optimizer and self._opt_state:
            self.optimizer.load_state_dict(self._opt_state)
        self.ema = None

    @torch.enable_grad()
    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()

        out = self.model(x)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        entropys = _softmax_entropy(logits)
        ids = torch.where(entropys < self.margin_e0)[0]

        if not ids.numel():
            return logits

        loss1 = entropys[ids].mean()
        loss1.backward()
        self.optimizer.first_step(zero_grad=True)

        # second forward at perturbed weights
        out2 = self.model(x)
        logits2 = out2[0] if isinstance(out2, (list, tuple)) else out2
        entropys2 = _softmax_entropy(logits2)
        ids2 = torch.where(entropys2 < self.margin_e0)[0]
        loss2 = entropys2[ids2].mean() if ids2.numel() else None

        if loss2 is not None and not torch.isnan(loss2):
            v = loss2.item()
            self.ema = v if self.ema is None else 0.9 * self.ema + 0.1 * v
            loss2.backward()

        self.optimizer.second_step(zero_grad=True)

        if self.ema is not None and self.ema < self.reset_constant_em:
            self._reset()

        return logits2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.model(x)
        logits = None
        for _ in range(self.steps):
            logits = self._adapt(x)
        return logits
