from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

from .tome_extensions import apply_tome_to_model


def _method_tome(cfg):
    """ToMe takes its depth profile from method.schedule, which defaults to the flat one."""
    return replace(cfg.method, schedule=str(getattr(cfg.method, "schedule", "constant") or "constant"))




















METHOD_BUILDERS: Dict[str, Callable] = {
    "tome": _method_tome,
}

BASELINE_MODEL_METHODS = {
    "atc",
    "ats",
    "evit",
    "pitome",
}

TOME_FAMILY_METHODS = {
    "tome",
}


def resolve_method_cfg(cfg):
    name = cfg.method.name.lower()
    if name in BASELINE_MODEL_METHODS:
        return cfg.method
    if name not in METHOD_BUILDERS:
        raise ValueError(f"Unsupported method: {cfg.method.name}")
    return METHOD_BUILDERS[name](cfg)


def patch_model_for_method(model, cfg):
    method_cfg = resolve_method_cfg(cfg)
    method_name = method_cfg.name.lower()

    if method_name in TOME_FAMILY_METHODS and getattr(method_cfg, "reduction_loc", []):
        print(
            "[Warning] method.reduction_loc is ignored for ToMe-family methods in this runner. "
            "ToMe schedule is applied across all transformer blocks via method.schedule/total_r."
        )

    if method_cfg.name.lower() in BASELINE_MODEL_METHODS:
        return model, method_cfg
    return apply_tome_to_model(model, method_cfg), method_cfg
