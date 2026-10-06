"""Configs for DomainNet-126 on one backbone finetuned on the real domain: the held-out real split
and the sketch, clipart and painting targets, each with no reduction, flat r_base=8 and late gamma=2.

    python scripts/experiments/natural_shift_domainnet/make_configs_dn.py <model_name> <tag> <checkpoint.pth>
    python scripts/experiments/sweep.py output/natural_shift_domainnet output/natural_shift_domainnet/<tag>_jobs.json
    python scripts/experiments/natural_shift_domainnet/stats_dn.py <tag> "<label>"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, SCHEDULES_R8, cell, load_base, write_jobs

if len(sys.argv) < 4:
    sys.exit(__doc__)
model, tag, checkpoint = sys.argv[1:4]
base = load_base("vits_const_r8.json", model_name=model, pretrained=False, num_classes=126, checkpoint_path=checkpoint)
cells = [cell(dom, "domainnet126", dom, "all", s, SCHEDULES_R8[s], base)
         for dom in ("real", "sketch", "clipart", "painting") for s in ("none", "flat", "late")]
write_jobs(REPO / "output" / "natural_shift_domainnet", tag, cells)
