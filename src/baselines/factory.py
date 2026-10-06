from __future__ import annotations

from argparse import Namespace
from functools import partial

import timm
import torch.nn as nn
import torch
from src.methods.schedules import (
    build_early_concentrated_schedule,
    build_exponential_schedule,
    build_late_concentrated_schedule,
)

from src.baselines.models import (
    ATCVisionTransformer,
    ATSVisionTransformer,
    EfficientVisionTransformer,
    PiToMeVisionTransformer,
)


BASELINE_METHODS = {"ats", "evit", "pitome", "atc"}



BASELINE_CLASS_MAP = {
    "atc": ATCVisionTransformer,
    "evit": EfficientVisionTransformer,
    "pitome": PiToMeVisionTransformer,
    "ats": ATSVisionTransformer,
}

SCALE_SPECS = {
    "tiny": {"embed_dim": 192, "num_heads": 3},
    "small": {"embed_dim": 384, "num_heads": 6},
    "base": {"embed_dim": 768, "num_heads": 12},
}

RATIO_BASELINES = {"evit", "pitome"}
COUNT_BASELINES = {"ats", "atc"}


def is_baseline_method(method_name: str) -> bool:
    return method_name.lower() in BASELINE_METHODS




def _extract_scale(model_name: str) -> str:
    if "tiny" in model_name:
        return "tiny"
    if "small" in model_name:
        return "small"
    if "base" in model_name:
        return "base"
    return "small"


def _baseline_args(cfg, num_classes: int) -> Namespace:
    method_name = cfg.method.name.lower()
    reduction_loc, keep_rate = _resolve_baseline_reduction_profile(cfg)
    args = Namespace(
        keep_rate=keep_rate,
        reduction_loc=reduction_loc,
        reduction_ratio=keep_rate,
        cluster_iters=cfg.method.cluster_iters,
        k_neighbors=cfg.method.k_neighbors,
        not_contiguous=cfg.method.not_contiguous,
        min_radius=cfg.method.min_radius,
        linkage=getattr(cfg.method, "linkage", "average"),
        cluster_backend=getattr(cfg.method, "cluster_backend", "sklearn"),
        proportional_attn=getattr(cfg.method, "proportional_attn", True),
        pitome_use_bsm_pitome=getattr(cfg.method, "pitome_use_bsm_pitome", False),
        distillation_type="none",
        num_classes=num_classes,
    )
    return args








def _resolve_baseline_reduction_profile(cfg) -> tuple[list[int], list[float] | list[int]]:
    schedule = getattr(cfg.method, "schedule", "constant").lower()
    method_name = cfg.method.name.lower()
    num_layers = 12
    init_tokens = 14 * 14
    total_budget = int(cfg.method.total_r) * num_layers
    if schedule == "constant":
        layer_r = [int(cfg.method.total_r)] * num_layers
    elif schedule == "late-concentrated":
        layer_r = build_late_concentrated_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            gamma=getattr(cfg.method, "schedule_gamma", 2.0),
        )
    elif schedule == "early-concentrated":
        layer_r = build_early_concentrated_schedule(
            total_r=total_budget,
            num_layers=num_layers,
            gamma=getattr(cfg.method, "schedule_gamma", 2.0),
        )
    elif schedule == "exponential":
        layer_r = build_exponential_schedule(total_budget, num_layers, getattr(cfg.method, "schedule_beta", 1.0))
    else:
        reduction_loc = list(getattr(cfg.method, "reduction_loc", []))
        keep_rate = list(getattr(cfg.method, "keep_rate", [0.7]))
        return reduction_loc, keep_rate

    # Bipartite matching pairs tokens, so a layer can remove at most half of what reaches it.
    remaining = init_tokens
    cumulative_counts: list[int] = []
    for r in layer_r:
        r = min(int(r), remaining // 2)
        remaining = max(1, remaining - r)
        cumulative_counts.append(remaining)

    # Ratio reducers reconstruct an integer token count as int(rate * init_tokens); half a token of
    # headroom keeps that from truncating a count down by one.
    _keep_epsilon = 0.5 / float(init_tokens)
    reduction_loc = list(range(num_layers))

    if method_name in COUNT_BASELINES:
        keep_values: list[int] = cumulative_counts
    elif method_name in RATIO_BASELINES:
        # A layer that removes nothing keeps every token, and a keep rate of 1.0 is what says so.
        keep_values = [min(1.0, count / float(init_tokens) + _keep_epsilon) for count in cumulative_counts]
    else:
        keep_values = list(getattr(cfg.method, "keep_rate", [0.7]))
        reduction_loc = list(getattr(cfg.method, "reduction_loc", []))

    return reduction_loc, keep_values


def create_baseline_model(cfg, num_classes: int):
    method_name = cfg.method.name.lower()
    if method_name not in BASELINE_METHODS:
        raise ValueError(f"Unsupported baseline method: {cfg.method.name}")

    scale = _extract_scale(cfg.model_name)
    spec = SCALE_SPECS[scale]
    cls = BASELINE_CLASS_MAP[method_name]

    model = cls(
        patch_size=16,
        embed_dim=spec["embed_dim"],
        depth=12,
        num_heads=spec["num_heads"],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        args=_baseline_args(cfg, num_classes),
    )

    if cfg.pretrained:
        base = timm.create_model(cfg.model_name, pretrained=True, num_classes=num_classes)
        model.load_state_dict(base.state_dict(), strict=False)

    model.eval()
    return model
