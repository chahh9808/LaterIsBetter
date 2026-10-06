"""Per-layer token counts and GFLOPs of the schedules in output/mae_gap/*.json, read back from the
running model with forward hooks rather than from the requested counts: ToMe removes at most half
of the tokens present at a layer, so a schedule can remove fewer tokens than it asks for.

    CKPT_DIR=checkpoints python scripts/experiments/mae_gap/token_audit.py
"""
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mae import MODELS, load_model, set_r, set_seed  # noqa: E402
from src.efficiency import estimate_gflops  # noqa: E402

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
for key, spec in MODELS.items():
    files = sorted(glob.glob(str(REPO / "output" / "mae_gap" / f"mae_{key}_g*.json")))
    if not files:
        continue
    model = load_model(spec, device)
    counts = []
    hooks = [b.register_forward_hook(lambda m, i, o: counts.append(o.shape[1])) for b in model.blocks]
    x = torch.randn(2, 3, 224, 224, device=device)
    print(f"\n{spec['arch']} (196 patches + class token)")
    seen = {}
    for f in files:
        d = json.load(open(f))
        for name in ("flat", "late"):
            label = "flat" if name == "flat" else f"late gamma={d['gamma']:g}"
            if label in seen:
                continue
            seen[label] = True
            counts.clear()
            set_r(model, d[name])
            with torch.no_grad():
                model(x)
            g = estimate_gflops(model, torch.randn(1, 3, 224, 224, device=device))
            print(f"  {label:14s} requested {sum(d[name]):4d}  removed {197 - counts[-1]:4d}  "
                  f"final tokens {counts[-1]:3d} (patches {counts[-1] - 1})  {g:.4f} GFLOPs")
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
