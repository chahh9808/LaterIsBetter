"""Count the FLOPs one adaptation step costs, per method and token budget (the cost appendix).

    python scripts/experiments/tta_cost/flops.py --backbone ViT-S --sched none --out output/tta_cost

Unlike measure.py this is deterministic: FlopCounterMode counts the matmuls and
convolutions the step actually issues, so a method's extra forward passes, its backward pass
and its auxiliary modules are all included, and a method that selects a subset of the batch
is counted on what it really propagated. Several batches are averaged because the selecting
methods (EATA, SAR) vary with the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from measure import METHODS, MODEL, SCHEDS, build_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(MODEL))
    ap.add_argument("--sched", required=True, choices=SCHEDS)
    ap.add_argument("--corruption", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
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
    need = args.warmup + args.batches
    staged = None
    results = []

    for method in [m for m in args.methods.split(",") if m]:
        cfg_dict = build_cfg(args.backbone, args.sched, method, out_dir, args.corruption)
        cfg_path = out_dir / f"fcfg_{args.backbone}_{args.sched}_{method}.json"
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))
        cfg = load_config(str(cfg_path))
        runner.set_seed(cfg.seed)

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
            del bundle

        model.eval()
        # The stream is online, so let the method reach its steady state before counting.
        for i in range(args.warmup):
            forward_with_method(model, staged[i].to(device, non_blocking=True), method_cfg)

        per_batch = []
        for i in range(args.warmup, need):
            x = staged[i].to(device, non_blocking=True)
            counter = FlopCounterMode(display=False)
            with counter:
                forward_with_method(model, x, method_cfg)
            per_batch.append(counter.get_total_flops())

        mean_flops = sum(per_batch) / len(per_batch)
        row = {"backbone": args.backbone, "schedule": args.sched, "method": method,
               "gflops_per_step": round(mean_flops / 1e9, 2),
               "gflops_per_image": round(mean_flops / 1e9 / 64, 3),
               "min_gflops": round(min(per_batch) / 1e9, 2),
               "max_gflops": round(max(per_batch) / 1e9, 2),
               "batches": args.batches, "batch_size": 64}
        results.append(row)
        print(json.dumps(row), flush=True)

        del model
        torch.cuda.empty_cache()

    (out_dir / f"flops_{args.backbone}_{args.sched}.json").write_text(json.dumps(results, indent=2))
    print("DONE", args.backbone, args.sched, flush=True)


if __name__ == "__main__":
    main()
