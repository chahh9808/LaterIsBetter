from __future__ import annotations

from typing import Dict

import os

import numpy as np
import torch

from .efficiency import estimate_realized_gflops


def _extract_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _extract_labels(targets):
    if isinstance(targets, dict):
        if "label" in targets:
            return targets["label"]
        if "labels" in targets:
            return targets["labels"]
    return targets


def _multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets > 0

    tp = (preds & targets).sum().item()
    fp = (preds & ~targets).sum().item()
    fn = (~preds & targets).sum().item()

    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


def _average_precision_binary(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    labels = labels > 0
    pos_count = int(labels.sum().item())
    if pos_count == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    ap = (precision * y).sum() / y.sum().clamp_min(1.0)
    return float(ap.item())


def _multilabel_map(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    targets = targets.float()

    aps = []
    for c in range(probs.shape[1]):
        ap = _average_precision_binary(probs[:, c], targets[:, c])
        if ap == ap:
            aps.append(ap)

    return float(sum(aps) / len(aps)) if aps else 0.0


@torch.no_grad()
def evaluate_top1(model, loader, device: torch.device, max_batches=None, method_forward=None, method_cfg=None, cfg=None, dump_path=None) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    ml_logits = []
    ml_targets = []
    gflops_cache = {}
    realized_gflops = []
    flags = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            continue

        inputs = inputs.to(device, non_blocking=True)
        targets = _extract_labels(targets)

        if not torch.is_tensor(targets):
            continue

        if targets.ndim > 1:
            outputs = method_forward(model, inputs, method_cfg) if method_forward is not None else model(inputs)
            if cfg is not None and cfg.efficiency.compute_gflops:
                value = estimate_realized_gflops(model, inputs, method_cfg, gflops_cache)
                if value is not None:
                    realized_gflops.append(float(value))
            logits = _extract_logits(outputs).detach().float().cpu()
            targets_ml = targets.detach().float().cpu()

            if logits.ndim == 2:
                while targets_ml.ndim > 2:
                    targets_ml = targets_ml.amax(dim=1)
                if targets_ml.shape == logits.shape:
                    ml_logits.append(logits)
                    ml_targets.append(targets_ml)
            continue

        targets = targets.to(device, non_blocking=True)
        outputs = method_forward(model, inputs, method_cfg) if method_forward is not None else model(inputs)
        if cfg is not None and cfg.efficiency.compute_gflops:
            value = estimate_realized_gflops(model, inputs, method_cfg, gflops_cache)
            if value is not None:
                realized_gflops.append(float(value))
        logits = _extract_logits(outputs)
        preds = torch.argmax(logits, dim=1)
        if dump_path:
            flags.append((preds == targets).cpu())

        correct += (preds == targets).sum().item()
        total += targets.numel()

    if dump_path and flags:

        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)

        np.save(dump_path, torch.cat(flags).numpy().astype(bool))

    if total > 0:
        top1 = (100.0 * correct / total)
        out = {"top1": top1, "num_samples": float(total)}
        if realized_gflops:
            out["gflops"] = float(sum(realized_gflops) / len(realized_gflops))
        return out

    if ml_logits:
        logits_all = torch.cat(ml_logits, dim=0)
        targets_all = torch.cat(ml_targets, dim=0)
        out = {
            "multilabel_map": _multilabel_map(logits_all, targets_all),
            "multilabel_micro_f1": _multilabel_micro_f1(logits_all, targets_all),
            "num_samples": float(targets_all.shape[0]),
        }
        if realized_gflops:
            out["gflops"] = float(sum(realized_gflops) / len(realized_gflops))
        return out

    return {"top1": 0.0, "num_samples": 0.0}
