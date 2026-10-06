from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import torch

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Lock the kernels so a (config, seed) gives one result: a long adaptation stream amplifies
    # cudnn's run-to-run differences into a different final accuracy.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        # warn_only: a few CUDA kernels have no deterministic implementation, and warning keeps
        # the rest of the run deterministic instead of raising on them.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-shift robust token-merging runner")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--wandb", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated wandb tags")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save final metrics as JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Finetuned weights to load (sets checkpoint_path)")
    parser.add_argument("--dump-preds", type=str, default=None, help=".npy path for per-image correctness")
    return parser.parse_args()


def _warmup_foa_source_stats(model, cfg, device, n_batches: int = 50) -> None:
    """Pre-compute layer-wise CLS feature mean/std for FOA on clean ImageNet."""
    from src.tta_library.foa import FOA
    if not isinstance(model, FOA):
        return
    if model.train_info is not None:
        return
    from src.datasets_local import resolve_dataset_roots
    in_root = resolve_dataset_roots(cfg.dataset_roots)["IN"]
    val_root = ""
    for candidate in [
        os.path.join(in_root, "Data", "CLS-LOC", "val"),
        os.path.join(in_root, "val"),
        in_root,
    ]:
        if os.path.isdir(candidate):
            val_root = candidate
            break
    if not val_root:
        print("[FOA] no clean source root found — skipping source-stat collection")
        return
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from src.data import _apply_model_transform
    ds = ImageFolder(root=val_root)
    _apply_model_transform(ds, cfg.model_name, input_size=224)
    # Fixed sampling seed (NOT cfg.seed): source stats are clean-data statistics,
    # identical across run seeds, and shared by all per-seed FOA runs via the
    # cached .pt — so the sample must be reproducible and seed-independent.
    g = torch.Generator().manual_seed(2020)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True, generator=g)
    print(f"[FOA] Computing source CLS-feature stats from clean ImageNet ({n_batches} batches)...")
    model.obtain_origin_stat(loader, n_batches=n_batches)
    print("[FOA] Source-stat pre-computation complete.")


def _warmup_eata_fisher(model, cfg, device, fisher_size: int = 2000) -> None:
    """Pre-compute Fisher diagonal from clean source data for EATA.

    Uses exactly `fisher_size` samples (official EATA default = 2000).
    """
    from src.tta_library.eata import EATA
    if not isinstance(model, EATA):
        return
    if model.fisher_alpha <= 0:
        return
    from src.datasets_local import resolve_dataset_roots
    in_root = resolve_dataset_roots(cfg.dataset_roots)["IN"]
    val_root = ""
    for candidate in [
        os.path.join(in_root, "Data", "CLS-LOC", "val"),
        os.path.join(in_root, "val"),
        in_root,
    ]:
        if os.path.isdir(candidate):
            val_root = candidate
            break
    if not val_root:
        return
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from src.data import _apply_model_transform
    ds = ImageFolder(root=val_root)
    _apply_model_transform(ds, cfg.model_name, input_size=224)
    g = __import__("torch").Generator().manual_seed(int(cfg.seed))
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                        pin_memory=True, generator=g)
    print(f"[EATA] Computing Fisher from {fisher_size} clean ImageNet samples...")
    model.compute_fisher(loader, fisher_size=fisher_size)
    print("[EATA] Fisher pre-computation complete.")


def main():
    args = parse_args()

    # Lazy imports keep --help functional even when heavy runtime deps are missing.
    from src.data import build_eval_data
    from src.eval import evaluate_top1
    from src.efficiency import collect_efficiency_metrics
    from src.logging_utils import init_wandb, log_metrics
    from src.methods import forward_with_method, patch_model_for_method
    from src.modeling import create_model
    from src.tta_library import build_tta_model
    from src.baselines.models.atc import warm_atc_rapids_clusterers

    overrides = {}

    if args.device is not None:
        overrides["device"] = args.device
    if args.method is not None:
        overrides["method"] = {"name": args.method}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_batches is not None:
        overrides.setdefault("evaluation", {})["max_batches"] = args.max_batches
    if args.dump_preds:
        overrides.setdefault("evaluation", {})["dump_predictions"] = args.dump_preds
    if args.checkpoint:
        overrides["checkpoint_path"] = args.checkpoint
    if args.wandb is not None:
        overrides.setdefault("logging", {})["use_wandb"] = args.wandb.lower() == "true"
    if args.wandb_project is not None:
        overrides.setdefault("logging", {})["project"] = args.wandb_project
    if args.wandb_group is not None:
        overrides.setdefault("logging", {})["group"] = args.wandb_group
    if args.wandb_run_name is not None:
        overrides.setdefault("logging", {})["run_name"] = args.wandb_run_name
    if args.wandb_tags is not None:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        overrides.setdefault("logging", {})["tags"] = tags

    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode

    cfg = load_config(args.config, overrides=overrides if overrides else None)
    set_seed(cfg.seed)

    method_name = cfg.method.name.lower()

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    data_bundle = None
    num_classes = cfg.num_classes if cfg.num_classes > 0 else None
    if num_classes is None:
        data_bundle = build_eval_data(cfg)
        num_classes = data_bundle.num_classes

    model = create_model(
        cfg,
        num_classes=num_classes,
    )

    model, method_cfg = patch_model_for_method(model, cfg)
    from src.datasets_local import IMAGENET_A_MASK, IMAGENET_R_MASK, ClassMaskedModel
    if cfg.dataset_name.lower() == "imagenet-r":
        model = ClassMaskedModel(model, IMAGENET_R_MASK)
    elif cfg.dataset_name.lower() == "imagenet-a":
        model = ClassMaskedModel(model, IMAGENET_A_MASK)
    model = build_tta_model(model, cfg)
    model = model.to(device)
    warm_atc_rapids_clusterers(model)
    _warmup_eata_fisher(model, cfg, device)
    _warmup_foa_source_stats(model, cfg, device)

    if data_bundle is None:
        data_bundle = build_eval_data(cfg)

    run = init_wandb(cfg)

    sample_batch = next(iter(data_bundle.loader))[0][:1]

    metrics_eff = collect_efficiency_metrics(model, sample_batch, cfg, method_cfg=method_cfg)
    if "gflops" in metrics_eff:
        metrics_eff["initial_gflops"] = metrics_eff["gflops"]
    metrics_eval = evaluate_top1(
        model,
        data_bundle.loader,
        device,
        max_batches=cfg.evaluation.max_batches,
        dump_path=cfg.evaluation.dump_predictions or None,
        method_forward=forward_with_method,
        method_cfg=method_cfg,
        cfg=cfg,
    )

    metrics = {
        **metrics_eff,
        **metrics_eval,
        "method": method_cfg.name,
        "dataset_name": cfg.dataset_name,
        "dataset_split": cfg.dataset_split,
        "corruption": str(cfg.corruption),
        "level": cfg.level,
        "seed": cfg.seed,
        "config": {
            "model": cfg.model_name,
            "tta": dataclasses.asdict(cfg.tta),
            "token_reduction": dataclasses.asdict(method_cfg),
        },
    }

    log_metrics(run, metrics)
    if run is not None:
        run.finish()

    print("Final metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
