from __future__ import annotations

import json
from typing import Callable, Dict, Optional, Tuple

import torch

from src.methods.schedules import (
    build_piecewise_increasing_schedule,
    build_late_concentrated_schedule,
    build_early_concentrated_schedule,
    build_exponential_schedule,
)

import importlib


def _tome_patch():
    """The ToMe timm patch module, imported on first use so the reducers that do not use ToMe run
    without it installed. The package attribute of the same name is apply_patch itself, hence import_module."""
    return importlib.import_module("tome.patch.timm")


def forward_with_method(model, inputs: torch.Tensor, method_cfg):
    setattr(model, "_last_realized_schedule_sequence", None)
    return model(inputs)


def apply_tome_to_model(model, method_cfg):
    _tome_patch().apply_patch(model, trace_source=False, prop_attn=True)

    import types
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None or getattr(attn, "_adaptive_entropy_wrapped", False):
            continue
        def wrapped_forward(self, x, size=None):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            if size is not None:
                attn_scores = attn_scores + size.log()[:, None, None, :, 0]
            attn_probs = self.attn_drop(attn_scores.softmax(dim=-1))
            out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out, k.mean(1)
        attn.forward = types.MethodType(wrapped_forward, attn)
        attn._adaptive_entropy_wrapped = True

    schedule = str(getattr(method_cfg, "schedule", "constant")).lower()
    num_layers = len(model.blocks)
    total_budget = int(method_cfg.total_r) * num_layers
    if schedule == "piecewise-increasing":
        model.r = build_piecewise_increasing_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            breakpoints=getattr(method_cfg, "schedule_breakpoints", []),
            weights=getattr(method_cfg, "schedule_weights", []),
        )
    elif schedule == "late-concentrated":
        model.r = build_late_concentrated_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            gamma=getattr(method_cfg, "schedule_gamma", 2.0),
        )
    elif schedule == "early-concentrated":
        model.r = build_early_concentrated_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            gamma=getattr(method_cfg, "schedule_gamma", 2.0),
        )
    elif schedule == "exponential":
        model.r = build_exponential_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            beta=method_cfg.schedule_beta,
        )
    elif schedule == "constant":
        model.r = method_cfg.total_r
    else:
        raise ValueError(f"Unsupported ToMe schedule: {method_cfg.schedule}")

    if isinstance(model.r, list):
        model._tome_base_r = list(model.r)
    else:
        model._tome_base_r = model.r
    return model
