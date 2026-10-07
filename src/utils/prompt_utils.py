from __future__ import annotations

import torch
import torch.nn as nn


def _get_inner_model(model: nn.Module) -> nn.Module:
    """Unwrap any TTA wrapper to get the base timm model."""
    inner = model
    while not hasattr(inner, "blocks") and hasattr(inner, "model"):
        inner = inner.model
    return inner


def _make_prompt_pre_hook(prompts_param: nn.Parameter):
    """Returns a forward_pre_hook that injects prompt tokens after the CLS token."""
    def hook(module, input):
        x = input[0]
        B = x.shape[0]
        p = prompts_param.to(x.device).expand(B, -1, -1)
        # Inject after CLS (pos 0), before all patch tokens (pos 1..N)
        return (torch.cat([x[:, :1], p, x[:, 1:]], dim=1),)
    return hook
