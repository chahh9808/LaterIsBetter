from __future__ import annotations

import torch.nn as nn


# SAR/DeYO: skip top 3 blocks (9-11) and final norm for ViT-12 (DeiT-S/B).
# "norm." carries the dot so it matches the final norm and not "norm1" / "norm2" inside a block.
_SAR_SKIP_SUBSTR = frozenset({"blocks.9", "blocks.10", "blocks.11", "norm."})
_SAR_SKIP_EXACT  = frozenset({"norm"})


def _collect_ln_params(model: nn.Module):
    """All LayerNorm/BN weight+bias."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for pn, p in m.named_parameters():
                if pn in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def _collect_sar_params(model: nn.Module):
    params, names = [], []
    for nm, m in model.named_modules():
        if any(s in nm for s in _SAR_SKIP_SUBSTR) or nm in _SAR_SKIP_EXACT:
            continue
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for pn, p in m.named_parameters():
                if pn in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def _configure_norm_only(model: nn.Module, collect_fn) -> list:
    for p in model.parameters():
        p.requires_grad_(False)
    params, _ = collect_fn(model)
    for p in params:
        p.requires_grad_(True)
    return params


def _find_classifier(model: nn.Module) -> nn.Linear:
    """The linear classifier at the end of the network, whatever wraps it.

    The runner may wrap the backbone (e.g. the class mask of ImageNet-R/A), so walk the ``.model``
    chain down to the module that owns the timm classifier. Methods that read or rewrite what the
    classifier sees hook this module.
    """
    module = model
    while not isinstance(getattr(module, "head", None), nn.Linear) and hasattr(module, "model"):
        module = module.model
    head = getattr(module, "head", None)
    if not isinstance(head, nn.Linear):
        raise ValueError("expected a model with a linear `head`; got " + type(head).__name__)
    return head
