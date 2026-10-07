"""Depth profiles for token reduction.

Every builder takes ``total_r`` as the number of tokens removed across the whole network
and returns the per-layer counts, which sum to it. The caller multiplies the per-layer
budget by the depth before calling, so a schedule and a flat baseline quoted at the same
``r_base`` remove the same number of tokens in total.

These are shared: the ToMe path and the baseline reducers build their schedules here, which
is what lets every reducer run the identical realized schedule.
"""

from __future__ import annotations

import torch

def _allocate_schedule_from_weights(total_r: int, weights: torch.Tensor) -> list[int]:
    if total_r <= 0 or weights.numel() == 0:
        return [0] * max(1, int(weights.numel()))

    weights = weights.clamp_min(0)
    if float(weights.sum().item()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()
    raw = weights * float(total_r)
    sched = torch.floor(raw).to(torch.int64)
    remainder = total_r - int(sched.sum().item())
    if remainder > 0:
        frac = raw - sched.float()
        topk = torch.topk(frac, k=remainder).indices
        sched[topk] += 1
    return sched.tolist()


def build_piecewise_increasing_schedule(
    total_r: int,
    num_layers: int,
    breakpoints: list[int],
    weights: list[float],
) -> list[int]:
    if total_r <= 0 or num_layers <= 0:
        return [0] * max(1, num_layers)

    bps = sorted(int(v) for v in breakpoints if 0 < int(v) < num_layers)
    while len(bps) < 2:
        default_bp = max(1, min(num_layers - 1, (len(bps) + 1) * num_layers // 3))
        if default_bp not in bps:
            bps.append(default_bp)
        bps = sorted(set(bps))
        if len(bps) == num_layers - 1:
            break
    bps = (bps + [num_layers])[:2]
    b1 = bps[0]
    b2 = bps[1] if len(bps) > 1 else num_layers

    phase_weights = [float(v) for v in weights] if weights else []
    if len(phase_weights) != 3:
        phase_weights = [1.0, 2.0, 4.0]

    layer_weights = torch.empty(num_layers, dtype=torch.float32)
    layer_weights[:b1] = phase_weights[0]
    layer_weights[b1:b2] = phase_weights[1]
    layer_weights[b2:] = phase_weights[2]
    return _allocate_schedule_from_weights(total_r, layer_weights)


def build_late_concentrated_schedule(total_r: int, num_layers: int, gamma: float = 2.0) -> list[int]:
    if total_r <= 0 or num_layers <= 0:
        return [0] * max(1, num_layers)

    positions = torch.linspace(1.0 / num_layers, 1.0, num_layers)
    layer_weights = torch.pow(positions, max(float(gamma), 1e-6))
    return _allocate_schedule_from_weights(total_r, layer_weights)


def build_early_concentrated_schedule(total_r: int, num_layers: int, gamma: float = 2.0) -> list[int]:
    """Mirror of late-concentrated: concentrates removal in early layers.

    Used as a controlled ablation to isolate whether the late-concentrated
    schedule benefit is specific to corruption (not just any non-flat schedule).
    """
    if total_r <= 0 or num_layers <= 0:
        return [0] * max(1, num_layers)

    positions = torch.linspace(1.0 / num_layers, 1.0, num_layers)
    layer_weights = torch.pow(positions.flip(0), max(float(gamma), 1e-6))
    return _allocate_schedule_from_weights(total_r, layer_weights)


def build_exponential_schedule(total_r: int, num_layers: int, beta: float = 1.0) -> list[int]:
    """Exponential late-heavy schedule: w_l = exp(beta * x_l).

    beta=0 → flat (constant), larger beta → more late-heavy.
    Total budget is conserved: sum(r_l) = total_r.
    """
    if total_r <= 0 or num_layers <= 0:
        return [0] * max(1, num_layers)

    x = torch.linspace(0.0, 1.0, num_layers)
    layer_weights = torch.exp(torch.tensor(float(beta)) * x)
    return _allocate_schedule_from_weights(total_r, layer_weights)
