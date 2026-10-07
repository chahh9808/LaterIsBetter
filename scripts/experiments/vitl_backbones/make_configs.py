"""Configs for the two 24-layer backbones of Table 4, AugReg ViT-L/16 and CLIP ViT-L/14.

The 12-layer backbones run at r_base=8. On 24 layers that per-layer rate would cut nearly twice the
compute, so each of these runs at the integer rate closest to the same 24-25% reduction, r_base=4
for ViT-L/16 and 5 for ViT-L/14, and the late budgets are the smallest whose GFLOPs stay at or
below flat's (vitl_calib.py reports these numbers).

    python scripts/experiments/vitl_backbones/make_configs.py
    python scripts/experiments/sweep.py output/vitl_backbones output/vitl_backbones/{augregL,clipL}_jobs.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CORRUPTIONS, REPO, cell, load_base, schedule, write_jobs

MODELS = {
    "augregL": dict(model="vit_large_patch16_224.augreg_in21k_ft_in1k", rb=4, g1=6, g2=8),
    "clipL": dict(model="vit_large_patch14_clip_224.openai_ft_in1k", rb=5, g1=8, g2=10),
}
for tag, spec in MODELS.items():
    base = load_base("vitb_const_r8.json", model_name=spec["model"])
    sched = {"none": schedule(0), "flat": schedule(spec["rb"]),
             "late_g1": schedule(spec["g1"], "late-concentrated", 1.0), "late_g2": schedule(spec["g2"], "late-concentrated", 2.0)}
    cells = [cell("clean", "imagenet", "val", "ori", s, sched[s], base) for s in sched]
    for i, c in enumerate(CORRUPTIONS):
        for s in ("flat", "late_g1", "late_g2"):
            cells.append(cell(f"inc_{c}", "imagenet", "c", str(i), s, sched[s], base))
    write_jobs(REPO / "output" / "vitl_backbones", tag, cells)
