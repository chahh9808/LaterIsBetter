"""The cost appendix table from the files measure.py and flops.py wrote.

    python scripts/experiments/tta_cost/summarize.py output/tta_cost

GFLOPs per step come from flops_<backbone>_<sched>.json; ms per step is each repeat's mean over its timed
steps, averaged over the repeats in cost_<backbone>_<sched>_<method>_rep*.json. cut is the fraction the
quantity shrank from full tokens (sched none) to the late schedule.
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import METHODS, MODEL  # noqa: E402

out = Path(sys.argv[1])
print("| backbone | method | GFLOPs full | GFLOPs late | cut | ms full | ms late | cut |")
for b in MODEL:
    gflops = {}
    for s in ("none", "late"):
        f = out / f"flops_{b}_{s}.json"
        gflops[s] = {r["method"]: r["gflops_per_step"] for r in json.load(open(f))} if f.exists() else {}
    for m in METHODS:
        ms = {}
        for s in ("none", "late"):
            reps = [json.load(open(f))[0]["ms_per_step"] for f in sorted(glob.glob(str(out / f"cost_{b}_{s}_{m}_rep*.json")))]
            ms[s] = st.mean(reps) if reps else None
        g = [gflops[s].get(m) for s in ("none", "late")]
        cells = [f"{g[0]:.0f}" if g[0] else "-", f"{g[1]:.0f}" if g[1] else "-",
                 f"{100 * (1 - g[1] / g[0]):.1f}%" if all(g) else "-",
                 f"{ms['none']:.1f}" if ms["none"] else "-", f"{ms['late']:.1f}" if ms["late"] else "-",
                 f"{100 * (1 - ms['late'] / ms['none']):.1f}%" if ms["none"] and ms["late"] else "-"]
        print(f"| {b} | {m} | " + " | ".join(cells) + " |")
