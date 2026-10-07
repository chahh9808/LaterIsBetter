# Adapted, with modifications, from https://github.com/Jhyun17/DeYO (MIT License, Copyright (c) 2024 Jonghyun Lee).
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.entropy import _softmax_entropy
from src.utils.param_utils import _collect_sar_params, _configure_norm_only


class DeYO(nn.Module):
    """DeYO: entropy + Patch-Level Prediction Difference filter (Lee et al., ICLR 2024).
    Official repo: https://github.com/Jhyun17/DeYO
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
        lr: float = 1e-3,
        momentum: float = 0.9,
        deyo_margin: float = 0.5 * math.log(1000),
        margin_e0: float = 0.4 * math.log(1000),
        plpd_threshold: float = 0.2,
    ):
        super().__init__()
        self.model = model
        self.steps = max(1, int(steps))
        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
        self.plpd_threshold = plpd_threshold

        params = _configure_norm_only(model, _collect_sar_params)
        self.optimizer = (
            torch.optim.SGD(params, lr=lr, momentum=momentum) if params else None
        )

    @staticmethod
    def _permute_patches(x: torch.Tensor) -> torch.Tensor:
        """Random 4×4 patch permutation (pure PyTorch, no einops)."""
        B, C, H, W = x.shape
        H4, W4 = (H // 4) * 4, (W // 4) * 4
        x_crop = x[:, :, :H4, :W4]
        ph, pw = H4 // 4, W4 // 4
        patches = x_crop.reshape(B, C, 4, ph, 4, pw).permute(0, 2, 4, 1, 3, 5).reshape(B, 16, C, ph, pw)
        idx = torch.argsort(torch.rand(B, 16, device=x.device), dim=-1)
        patches = patches[torch.arange(B, device=x.device).unsqueeze(-1), idx]
        x_perm = patches.reshape(B, 4, 4, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H4, W4)
        if H4 != H or W4 != W:
            x_perm = torch.nn.functional.interpolate(x_perm, size=(H, W), mode="bilinear", align_corners=False)
        return x_perm

    @torch.enable_grad()
    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out[0] if isinstance(out, (list, tuple)) else out

        entropys = _softmax_entropy(logits)
        ids1 = torch.where(entropys < self.deyo_margin)[0]
        if not ids1.numel():
            return logits

        entropys_sel = entropys[ids1]
        x_perm = self._permute_patches(x[ids1].detach())

        with torch.no_grad():
            out_p = self.model(x_perm)
            logits_p = out_p[0] if isinstance(out_p, (list, tuple)) else out_p

        probs = logits[ids1].softmax(1)
        probs_p = logits_p.softmax(1)
        cls1 = probs.argmax(1, keepdim=True)
        plpd = (torch.gather(probs, 1, cls1) - torch.gather(probs_p, 1, cls1)).squeeze(1)

        ids2 = torch.where(plpd > self.plpd_threshold)[0]
        entropys_sel = entropys_sel[ids2]
        plpd_sel = plpd[ids2]
        if not entropys_sel.numel():
            return logits

        coeff = (
            1.0 / torch.exp(entropys_sel.clone().detach() - self.margin_e0)
            + 1.0 / torch.exp(-plpd_sel.clone().detach())
        )
        loss = (entropys_sel * coeff).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.model(x)
        logits = None
        for _ in range(self.steps):
            logits = self._adapt(x)
        return logits
