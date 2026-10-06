from __future__ import annotations

import torch


def _softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
