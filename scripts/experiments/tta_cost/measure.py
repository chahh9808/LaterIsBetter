"""Wall-clock time and peak memory of one adaptation step, per method and token budget (the cost appendix).

One process handles one (backbone, schedule) pair and loops over --methods, so the ImageNet-C batches are
read from disk once. Only the model call is timed: the batches are staged in pinned memory beforehand and
the loop synchronizes around the call itself, so neither disk nor host-to-device transfer enters the
number. Deterministic kernels are off unless --deterministic is given; the reported times were measured
with them off, one process per (backbone, schedule, method), two repeats told apart by --tag:

    python scripts/experiments/tta_cost/measure.py --backbone ViT-S --sched none --methods tent --tag _tent_rep1 --out output/tta_cost
    python scripts/experiments/tta_cost/summarize.py output/tta_cost

--sched is none (197 tokens in every block), flat (r_base=8) or late (gamma=2 at flat's GFLOPs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import torch  # noqa: E402

from common import load_base, schedule  # noqa: E402
from launch_table import BASE, LR, METHODS6 as METHODS, late  # noqa: E402

# the Table 6 backbones; the learning rates are Table 6's, so each cell is the method as it is run there
MODEL = {"ViT-S": ("vit_small_patch16_224", "S"), "ViT-B": ("vit_base_patch16_224", "B")}
SCHEDS = ("none", "flat", "late")


def method_block(sched, scale):
    if sched == "none":
        return schedule(0)
    if sched == "flat":
        return schedule(8)
    return late(scale, 8, 2.0)


def build_cfg(backbone, sched, method, out_dir, corruption_index):
    name, scale = MODEL[backbone]
    base = dict(load_base(BASE[backbone], model_name=name))
    base["method"] = method_block(sched, scale)
    base["dataset_name"] = "imagenet"
    base["dataset_split"] = "c"
    base["corruption"] = str(corruption_index)
    base["level"] = 5
    base["seed"] = 42
    tta = {"enabled": True, "method": method, "steps": 1,
           "foa_source_stats_path": str(out_dir / f"foa_{name}_{sched}.pt")}
    if LR.get((backbone, method)) is not None:
        tta["lr"] = LR[(backbone, method)]
    base["tta"] = tta
    base["evaluation"] = {"batch_size": 64, "workers": 8, "max_batches": None}
    base.setdefault("efficiency", {})
    base["efficiency"] = {**base["efficiency"], "compute_gflops": False,
                          "compute_gpu_latency": False, "compute_cpu_latency": False}
    base.setdefault("logging", {})
    base["logging"] = {**base["logging"], "use_wandb": False}
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(MODEL))
    ap.add_argument("--sched", required=True, choices=SCHEDS)
    ap.add_argument("--corruption", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--deterministic", action="store_true", help="keep the deterministic kernels main.py selects")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.config import load_config
    from src.data import build_eval_data
    from src.methods import forward_with_method, patch_model_for_method
    from src.modeling import create_model
    from src.tta_library import build_tta_model
    import main as runner

    device = torch.device("cuda")
    need = args.warmup + args.iters

    # One read of the stream, staged in pinned memory, shared by every method in this process.
    staged = None
    results = []

    for method in [m for m in args.methods.split(",") if m]:
        cfg_dict = build_cfg(args.backbone, args.sched, method, out_dir, args.corruption)
        cfg_path = out_dir / f"cfg_{args.backbone}_{args.sched}_{method}.json"
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))
        cfg = load_config(str(cfg_path))
        runner.set_seed(cfg.seed)
        if not args.deterministic:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        model = create_model(cfg, num_classes=1000)
        model, method_cfg = patch_model_for_method(model, cfg)
        model = build_tta_model(model, cfg)
        model = model.to(device)
        runner._warmup_eata_fisher(model, cfg, device)
        runner._warmup_foa_source_stats(model, cfg, device)

        if staged is None:
            bundle = build_eval_data(cfg)
            staged = []
            for batch in bundle.loader:
                staged.append(batch[0].pin_memory())
                if len(staged) >= need:
                    break
            if len(staged) < need:
                raise SystemExit(f"stream gave {len(staged)} batches, need {need}")
            del bundle
            torch.cuda.empty_cache()

        model.eval()
        for i in range(args.warmup):
            forward_with_method(model, staged[i].to(device, non_blocking=True), method_cfg)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        per_step = []
        for i in range(args.warmup, need):
            x = staged[i].to(device, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            forward_with_method(model, x, method_cfg)
            torch.cuda.synchronize()
            per_step.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

        per_step.sort()
        ms = 1000.0 * sum(per_step) / len(per_step)
        row = {"backbone": args.backbone, "schedule": args.sched, "method": method,
               "ms_per_step": round(ms, 2),
               "ms_median": round(1000.0 * per_step[len(per_step) // 2], 2),
               "img_per_s": round(64.0 / (ms / 1000.0), 1),
               "peak_mem_gib": round(peak, 2),
               "iters": args.iters, "batch_size": 64}
        results.append(row)
        print(json.dumps(row), flush=True)

        del model
        torch.cuda.empty_cache()

    stem = f"cost_{args.backbone}_{args.sched}{args.tag}"
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=2))
    print("DONE", args.backbone, args.sched, flush=True)


if __name__ == "__main__":
    main()
