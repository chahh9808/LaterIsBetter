"""Operating points for backbones whose depth or token count differs from the 12-layer models.

For each candidate per-layer rate the script reports the flat GFLOPs and its compute reduction, and
for gamma=1 and gamma=2 the smallest integer late budget whose GFLOPs stay at or below flat's, all
through the repository's own ToMe path.

    python scripts/experiments/vitl_backbones/vitl_calib.py [model_name ...]

Writes output/vitl_backbones/calibration.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import timm
import torch

from src.config import MethodConfig
from src.efficiency import estimate_gflops
from src.methods.schedules import build_late_concentrated_schedule
from src.methods.tome_extensions import apply_tome_to_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = sys.argv[1:] or ["vit_large_patch16_224.augreg_in21k_ft_in1k", "vit_large_patch14_clip_224.openai_ft_in1k"]


def gflops(model, x):
    return estimate_gflops(model, x)


def set_r(model, r):
    model.r = list(map(int, r))
    model._tome_base_r = list(map(int, r))


out = {}
for name in MODELS:
    cfg = MethodConfig(name="tome", total_r=0, schedule="constant")
    model = apply_tome_to_model(timm.create_model(name, pretrained=True, num_classes=1000).eval(), cfg).to(DEVICE)
    depth, tokens = len(model.blocks), model.patch_embed.num_patches
    x = torch.randn(1, 3, *model.patch_embed.img_size, device=DEVICE)
    set_r(model, [0] * depth)
    full = gflops(model, x)
    print(f"\n{name}: {depth} layers, {tokens} patches, {full:.3f} GFLOPs uncompressed")
    rows = {}
    for rb in range(2, 9):
        set_r(model, [rb] * depth)
        flat = gflops(model, x)
        late = {}
        for gamma in (1.0, 2.0):
            for total_r in range(rb, 4 * rb + 20):
                sched = build_late_concentrated_schedule(total_r * depth, depth, gamma)
                set_r(model, sched)
                g = gflops(model, x)
                if g <= flat:
                    late[str(gamma)] = {"total_r": total_r, "gflops": g, "tokens_removed": sum(sched)}
                    break
        rows[rb] = {"flat_gflops": flat, "cut_percent": 100 * (1 - flat / full), "late": late}
        l1, l2 = late["1.0"], late["2.0"]
        print(f"  r_base={rb}: flat {flat:7.3f} ({rows[rb]['cut_percent']:5.2f}% cut) | g1 total_r={l1['total_r']:2d} ({l1['gflops']:7.3f}) "
              f"| g2 total_r={l2['total_r']:2d} ({l2['gflops']:7.3f})")
    out[name] = {"depth": depth, "patches": tokens, "full_gflops": full, "rates": rows}
    del model
    torch.cuda.empty_cache()
dest = REPO / "output" / "vitl_backbones" / "calibration.json"
dest.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dest, "w"), indent=2)
print(f"\nwrote {dest}")
