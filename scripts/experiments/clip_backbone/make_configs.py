"""Configs for the CLIP ViT-B/16 backbone: clean ImageNet (no reduction, flat, late) and the 15
ImageNet-C corruptions at the main rate (flat, late gamma=1, late gamma=2) and the aggressive rate.

    python scripts/experiments/clip_backbone/make_configs.py [model_name] [tag]
    python scripts/experiments/sweep.py output/clip_backbone output/clip_backbone/<tag>_jobs.json
    python scripts/experiments/clip_backbone/table4_clip.py [tag]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CORRUPTIONS, REPO, SCHEDULES_R12, SCHEDULES_R8, cell, load_base, write_jobs

model = sys.argv[1] if len(sys.argv) > 1 else "vit_base_patch16_clip_224.openai_ft_in1k"
tag = sys.argv[2] if len(sys.argv) > 2 else "clipb_openai"
base = load_base("vitb_const_r8.json", model_name=model)
cells = [cell("clean", "imagenet", "val", "ori", s, SCHEDULES_R8[s], base) for s in ("none", "flat", "late")]
for i, c in enumerate(CORRUPTIONS):
    for s, m in list(SCHEDULES_R8.items())[1:] + list(SCHEDULES_R12.items()):
        cells.append(cell(f"inc_{c}", "imagenet", "c", str(i), s, m, base))
write_jobs(REPO / "output" / "clip_backbone", tag, cells)
