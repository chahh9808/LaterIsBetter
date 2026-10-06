from __future__ import annotations

from typing import Any, Dict


def init_wandb(cfg):
    if not cfg.logging.use_wandb:
        return None
    try:
        import wandb
    except Exception:
        print("wandb not available, continuing without wandb logging")
        return None

    run = wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        group=cfg.logging.group,
        tags=cfg.logging.tags,
        config={
            "seed": cfg.seed,
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "dataset_split": cfg.dataset_split,
            "method": cfg.method.__dict__,
            "tta": cfg.tta.__dict__,
            "evaluation": cfg.evaluation.__dict__,
            "efficiency": cfg.efficiency.__dict__,
        },
    )
    return run


def log_metrics(run, metrics: Dict[str, Any]):
    if run is None:
        print(metrics)
        return
    run.log(metrics)
