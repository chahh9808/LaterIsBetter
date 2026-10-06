"""MAE-pretrained ViT-B and ViT-L with global-average-pooling readout on ImageNet-C: the flat
schedule against the late-concentrated schedule at matched GFLOPs.

    CKPT_DIR=checkpoints python scripts/experiments/mae_gap/run_mae.py --model vitb --gamma 2
    CKPT_DIR=checkpoints python scripts/experiments/mae_gap/run_mae.py --model vitl --gamma 1

The weights are the official finetuned MAE checkpoints, mae_finetuned_vit_base.pth and
mae_finetuned_vit_large.pth (checkpoints/MANIFEST.json). ViT-B runs at r_base=8 and ViT-L at
r_base=4 over its 24 layers, a compute cut of about 25% in both cases; the late budget is the
smallest whose GFLOPs stay at or below flat's. The models were finetuned with the ImageNet default
normalization, which is applied here; the corruption suite is pre-rendered at 224x224.
Writes output/mae_gap/mae_<model>_g<gamma>.json. --no-eval reports the schedules only.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import timm
import tome
import torch
from tome.utils import parse_r
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.datasets_local import CORRUPTIONS, resolve_dataset_roots  # noqa: E402
from src.efficiency import estimate_gflops  # noqa: E402
from src.methods.schedules import _allocate_schedule_from_weights  # noqa: E402

MODELS = {"vitb": dict(arch="vit_base_patch16_224", depth=12, r_base=8, ckpt="mae_finetuned_vit_base.pth", batch=200),
          "vitl": dict(arch="vit_large_patch16_224", depth=24, r_base=4, ckpt="mae_finetuned_vit_large.pth", batch=128)}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(spec, device):
    model = timm.create_model(spec["arch"], pretrained=False, num_classes=1000, global_pool="avg").eval()
    path = Path(os.environ.get("CKPT_DIR", "checkpoints")) / spec["ckpt"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    tome.patch.mae(model, prop_attn=True)

    def forward(self, x):
        self._tome_info["r"] = parse_r(len(self.blocks), self.r)
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        return self.head(self.forward_features(x))

    model.forward = types.MethodType(forward, model)
    return model.to(device)


def set_r(model, r):
    model.r = [int(x) for x in r]
    model._tome_base_r = [int(x) for x in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    spec = MODELS[args.model]
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(spec, device)
    depth = len(model.blocks)
    x = torch.randn(1, 3, 224, 224, device=device)
    flat = [spec["r_base"]] * depth
    set_r(model, flat)
    flat_gflops = estimate_gflops(model, x)
    weights = torch.pow(torch.linspace(1.0 / depth, 1.0, depth), max(args.gamma, 1e-6))
    late = None
    for budget in range(6, 40 * depth):
        candidate = _allocate_schedule_from_weights(budget, weights)
        set_r(model, candidate)
        if estimate_gflops(model, x) <= flat_gflops:
            late = candidate
            break
    set_r(model, late)
    late_gflops = estimate_gflops(model, x)
    report = {"model": spec["arch"], "readout": "global average pooling", "gamma": args.gamma, "r_base": spec["r_base"],
              "flat": flat, "late": late, "gflops": {"flat": flat_gflops, "late": late_gflops}, "results": {}}
    print(f"{args.model} flat r{spec['r_base']} {flat_gflops:.4f} GFLOPs | late gamma={args.gamma:g} budget {sum(late)} {late_gflops:.4f} GFLOPs", flush=True)
    out = REPO / "output" / "mae_gap" / f"mae_{args.model}_g{args.gamma:g}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.no_eval:
        json.dump(report, open(out, "w"), indent=2)
        return
    root = resolve_dataset_roots({})["IN-C"]
    transform = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    t0 = time.time()
    for corruption in CORRUPTIONS:
        loader = DataLoader(ImageFolder(f"{root}/{corruption}/5", transform=transform), batch_size=spec["batch"],
                            num_workers=10, shuffle=False, pin_memory=True)
        correct = {"flat": 0, "late": 0}
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                for name, sched in (("flat", flat), ("late", late)):
                    set_r(model, sched)
                    correct[name] += (model(xb).argmax(1).cpu() == yb).sum().item()
                total += len(yb)
        acc = {k: 100.0 * v / total for k, v in correct.items()}
        report["results"][corruption] = acc
        print(f"[{time.time() - t0:6.0f}s] {corruption:18s} n={total} flat {acc['flat']:.2f} late {acc['late']:.2f} ({acc['late'] - acc['flat']:+.2f})", flush=True)
        json.dump(report, open(out, "w"), indent=2)
    mean = {k: statistics.mean(v[k] for v in report["results"].values()) for k in ("flat", "late")}
    print(f"\nmean over {len(report['results'])} corruptions: flat {mean['flat']:.2f} late {mean['late']:.2f} ({mean['late'] - mean['flat']:+.2f})")


if __name__ == "__main__":
    main()
