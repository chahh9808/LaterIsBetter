"""Marginal GFLOPs saved by removing R tokens at one block (the compute curve of the mechanism figure).

S(l) = GFLOPs(unreduced) - GFLOPs(R tokens merged at block l only), counted with THOP on DeiT-S. Writes
output/mechanism/marginal_gflops.json with S_thop (per block), g0 (unreduced GFLOPs) and R.

    python scripts/experiments/mechanism/marginal_gflops.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import torch  # noqa: E402

from common import load_base  # noqa: E402
from src.config import _from_dict  # noqa: E402
from src.efficiency import estimate_gflops  # noqa: E402
from src.methods import patch_model_for_method  # noqa: E402
from src.modeling import create_model  # noqa: E402

N_BLOCKS = 12
parser = argparse.ArgumentParser()
parser.add_argument("--r", type=int, default=16)
parser.add_argument("--out", default="output/mechanism/marginal_gflops.json")
args = parser.parse_args()

cfg_dict = load_base("const_r8.json", model_name="deit_small_patch16_224", num_classes=1000, pretrained=False)
cfg_dict["method"] = {"name": "tome", "total_r": 0, "schedule": "constant", "schedule_gamma": 1.0}
cfg = _from_dict(cfg_dict)
model = create_model(cfg, 1000)
model, _ = patch_model_for_method(model, cfg)
model = model.eval()
sample = torch.randn(1, 3, 224, 224)

model.r = [0] * N_BLOCKS
g0 = float(estimate_gflops(model, sample))
saved = []
for l in range(N_BLOCKS):
    r = [0] * N_BLOCKS
    r[l] = args.r
    model.r = r
    saved.append(g0 - float(estimate_gflops(model, sample)))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({"S_thop": saved, "g0": g0, "R": args.r}, open(args.out, "w"))
print(f"g0={g0:.4f}  S={[round(s, 4) for s in saved]}")
print(f"wrote {args.out}")
