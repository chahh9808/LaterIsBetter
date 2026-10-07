"""Flat versus late on the four ImageNet-C corruptions outside the standard fifteen (the appendix on them).

    python scripts/experiments/extra_corruptions/run_extra.py --out output/extra_corruptions

The loader's corruption list stops at the fifteen the benchmark reports, so this script reads the four extra
directories, <DATA_IN_C>/<corruption>/<level>/<wnid>/, itself. Everything else, the model, the schedule
patch and the evaluation, is the same code main.py runs; the schedules are the main operating point,
DeiT-S at r_base=8 and gamma=2 at flat's GFLOPs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from common import load_base, schedule  # noqa: E402

EXTRA = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="DeiT-S")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batches", type=int, default=None, help="stop early, for a smoke test")
    args = ap.parse_args()

    from src.config import load_config
    from src.data import _apply_model_transform
    from src.datasets_local import resolve_dataset_roots
    from src.eval import evaluate_top1
    from src.methods import forward_with_method, patch_model_for_method
    from src.modeling import create_model
    import main as runner

    MODEL = {"DeiT-S": ("deit_small_patch16_224", "const_r8.json")}
    name, base_cfg = MODEL[args.backbone]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(resolve_dataset_roots({})["IN-C"])
    device = torch.device("cuda")

    # the two schedules of the main operating point, exactly as the launcher builds them
    ISO = json.load(open(REPO / "configs" / "iso_gflops_gamma_table.json"))["table"]
    blocks = {"flat": schedule(8),
              "late": schedule(ISO["S"]["8"]["2.0"]["total_r"], "late-concentrated", 2.0)}

    results = {}
    for key, block in blocks.items():
        cfg_dict = dict(load_base(base_cfg, model_name=name))
        cfg_dict["method"] = block
        cfg_dict["tta"] = {"enabled": False}
        cfg_dict["evaluation"] = {"batch_size": args.batch_size, "workers": 8, "max_batches": None}
        cfg_dict["efficiency"] = {**cfg_dict.get("efficiency", {}), "compute_gflops": False,
                                  "compute_gpu_latency": False, "compute_cpu_latency": False}
        cfg_path = out / f"cfg_{args.backbone}_{key}.json"
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))
        cfg = load_config(str(cfg_path))
        runner.set_seed(cfg.seed)

        model = create_model(cfg, num_classes=1000)
        model, method_cfg = patch_model_for_method(model, cfg)
        model = model.to(device).eval()

        for corr in EXTRA:
            d = root / corr / str(args.level)
            ds = ImageFolder(root=str(d))
            _apply_model_transform(ds, cfg.model_name, input_size=224)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
            m = evaluate_top1(model, loader, device, max_batches=args.max_batches, method_forward=forward_with_method,
                              method_cfg=method_cfg, cfg=cfg)
            results[(key, corr)] = m["top1"]
            print(f"{key:5s} {corr:14s} top1 {m['top1']:6.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    rows = [{"corruption": c, "flat": results[("flat", c)], "late": results[("late", c)],
             "delta": round(results[("late", c)] - results[("flat", c)], 2)} for c in EXTRA]
    (out / "extra_corruptions.json").write_text(json.dumps(rows, indent=2))
    print("\ncorruption        flat    late   delta")
    for r in rows:
        print(f"{r['corruption']:14s} {r['flat']:7.2f} {r['late']:7.2f} {r['delta']:+7.2f}")
    print(f"{'mean':14s} {sum(r['flat'] for r in rows)/4:7.2f} {sum(r['late'] for r in rows)/4:7.2f} "
          f"{sum(r['delta'] for r in rows)/4:+7.2f}")


if __name__ == "__main__":
    main()
