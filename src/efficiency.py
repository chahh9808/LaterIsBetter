from __future__ import annotations

import time
from typing import Dict, Optional

import torch

SCHEDULE_ATTRS = (
    "_last_likelihood_schedule",
    "_last_ambiguity_schedule",
    "_last_risk_schedule",
    "_last_safety_schedule",
    "_last_stream_schedule",
)


def _get_model_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _skip_gflops_for_method(method_cfg) -> bool:
    if method_cfg is None:
        return False
    method_name = getattr(method_cfg, "name", "").lower()
    cluster_backend = getattr(method_cfg, "cluster_backend", "").lower()
    # Profiling runs a synthetic forward, which the RAPIDS clustering backend does not support.
    return method_name == "atc" and cluster_backend == "rapids"


def _custom_gflops_for_method(model, sample, method_cfg):
    if method_cfg is None:
        return None
    method_name = getattr(method_cfg, "name", "").lower()
    if method_name != "pitome":
        return None

    sample = sample.to(_get_model_device(model))
    was_training = model.training
    model.eval()
    _ = model(sample)
    model.train(was_training)

    value = getattr(model, "_last_effective_gflops", None)
    return float(value) if value is not None else None


def estimate_gflops(model, sample):
    try:
        from thop import profile
    except Exception:
        return None

    sample = sample.to(_get_model_device(model))
    macs, _ = profile(model, inputs=(sample,), verbose=False)
    return float(macs) / 1e9


def estimate_effective_gflops(model, sample, method_cfg):
    if _skip_gflops_for_method(method_cfg):
        return None
    custom_value = _custom_gflops_for_method(model, sample, method_cfg)
    if custom_value is not None:
        return custom_value
    sample = sample.to(_get_model_device(model))
    return estimate_gflops(model, sample)


def extract_schedule_key(model) -> tuple[int, ...]:
    for attr in SCHEDULE_ATTRS:
        value = getattr(model, attr, None)
        if isinstance(value, list) and value:
            return tuple(int(v) for v in value)
    value = getattr(model, "r", None)
    if isinstance(value, list) and value:
        return tuple(int(v) for v in value)
    return tuple()


def extract_schedule_sequence(model) -> list[tuple[int, ...]]:
    sequence = getattr(model, "_last_realized_schedule_sequence", None)
    if isinstance(sequence, list) and sequence:
        out = []
        for item in sequence:
            if isinstance(item, (list, tuple)) and item:
                out.append(tuple(int(v) for v in item))
        if out:
            return out
    key = extract_schedule_key(model)
    return [key] if key else []


def estimate_realized_gflops(
    model,
    sample,
    method_cfg,
    cache: Optional[dict[tuple[int, ...], float]] = None,
):
    if _skip_gflops_for_method(method_cfg):
        return None

    total = 0.0
    seen = False
    for key in extract_schedule_sequence(model):
        if cache is not None and key in cache:
            total += cache[key]
            seen = True
            continue
        value = estimate_effective_gflops(model, sample[:1], method_cfg)
        if value is None:
            continue
        if cache is not None:
            cache[key] = value
        total += value
        seen = True
    return total if seen else None


@torch.no_grad()
def measure_cpu_latency(model, sample, warmup: int = 10, iters: int = 50) -> float:
    orig_device = _get_model_device(model)
    was_training = model.training

    cpu_model = model.to("cpu").eval()
    sample = sample.to("cpu")

    for _ in range(warmup):
        _ = cpu_model(sample)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cpu_model(sample)
    t1 = time.perf_counter()

    model.to(orig_device)
    model.train(was_training)

    return (t1 - t0) * 1000.0 / iters


@torch.no_grad()
def measure_gpu_latency(model, sample, warmup: int = 20, iters: int = 100) -> float:
    if not torch.cuda.is_available():
        return float("nan")

    orig_device = _get_model_device(model)
    was_training = model.training

    model = model.to("cuda").eval()
    sample = sample.to("cuda")

    for _ in range(warmup):
        _ = model(sample)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(sample)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    model.to(orig_device)
    model.train(was_training)

    return (t1 - t0) * 1000.0 / iters


def collect_efficiency_metrics(model, sample, cfg, method_cfg=None) -> Dict[str, float]:
    out = {}

    if cfg.efficiency.compute_cpu_latency:
        out["cpu_latency_ms"] = measure_cpu_latency(
            model,
            sample,
            warmup=cfg.efficiency.cpu_warmup,
            iters=cfg.efficiency.cpu_iters,
        )

    if cfg.efficiency.compute_gpu_latency:
        out["gpu_latency_ms"] = measure_gpu_latency(
            model,
            sample,
            warmup=cfg.efficiency.gpu_warmup,
            iters=cfg.efficiency.gpu_iters,
        )

    if cfg.efficiency.compute_gflops:
        gflops = estimate_effective_gflops(model, sample, method_cfg)
        if gflops is not None:
            out["gflops"] = gflops

    out["num_parameters_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    return out
