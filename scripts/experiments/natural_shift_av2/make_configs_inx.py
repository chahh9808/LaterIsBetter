"""Configs for ImageNet-A, ImageNet-V2, ImageNet-R and ImageNet-Sketch on one backbone: no reduction, flat r_base=8, late gamma=2.

    python scripts/experiments/natural_shift_av2/make_configs_inx.py [model_name] [tag]
    python scripts/experiments/sweep.py output/natural_shift_av2 output/natural_shift_av2/<tag>_jobs.json
    python scripts/experiments/natural_shift_av2/stats_av2.py <tag> "<label>"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, SCHEDULES_R8, cell, load_base, write_jobs

model = sys.argv[1] if len(sys.argv) > 1 else "vit_small_patch16_224"
tag = sys.argv[2] if len(sys.argv) > 2 else "vits_inx"
base = load_base("vits_const_r8.json", model_name=model)
cells = [cell(ds, ds, "val", "all", s, SCHEDULES_R8[s], base) for ds in ("imagenet-a", "imagenet-v2", "imagenet-r", "imagenet-sketch") for s in ("none", "flat", "late")]
write_jobs(REPO / "output" / "natural_shift_av2", tag, cells)
