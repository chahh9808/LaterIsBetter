"""Print a table of the paper from the metrics that scripts/experiments/sweep.py wrote.

    python scripts/summarize_table.py <target> <out_dir>

target is the scripts/launch_table.py target whose jobs file is in out_dir: table1, table2, table3, table4,
table6, table6_full, nabirds, uncompressed, placement, shapes or shiftsuites. Each cell is the mean top-1 over the corruptions of the
target (Table 6: also over the three seeds, with their standard deviation); delta is late minus flat at the
same rate, severity, reducer, method or suite.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RATES3 = {"DeiT-S": {4: (2.0,), 6: (1.0, 2.0), 8: (1.0, 2.0), 10: (1.0,)}, "DeiT-B": {4: (2.0,), 6: (2.0,), 8: (2.0,)}}
RATES6 = {"ViT-S": (4, 8), "ViT-B": (4, 8)}
METHODS6 = ("tent", "eata", "sar", "foa", "spa", "neo")
SEEDS = (42, 43, 44)

table, out = sys.argv[1], Path(sys.argv[2])
runs = defaultdict(list)
for jobs_file in out.glob("*_jobs.json"):
    for j in json.load(open(jobs_file)):
        f = out / f"{j['tag']}_{j['dataset']}_{j['schedule']}.json"
        if not f.exists():
            continue
        parts = j["dataset"].split("_")
        group = "_".join(parts[:2]) if table == "shiftsuites" else parts[0] if table in ("table3", "table4", "table6", "table6_full", "nabirds") else ""
        runs[(group, j["schedule"])].append(json.load(open(f))["top1"])


def mean(group, sched):
    v = runs.get((group, sched))
    return st.mean(v) if v else None


def fmt(x, flat=None):
    if x is None:
        return "-"
    return f"{x:.2f}" if flat is None else f"{x:.2f} ({x - flat:+.2f})"


def row(label, flat, lates):
    print(f"| {label} | {fmt(flat)} | " + " | ".join(fmt(x, flat) for x in lates) + " |")


if table == "table1":
    gammas = (0.5, 1, 1.5, 2, 2.5, 3)
    print("| r_base | flat | " + " | ".join(f"gamma={g:g}" for g in gammas) + " |")
    for r in (2, 4, 6, 8, 10, 12):
        row(f"r{r}", mean("", f"r{r}_flat"), [mean("", f"r{r}_g{g:g}") for g in gammas])
elif table == "table2":
    print("| severity | flat | late gamma=1 | late gamma=2 |")
    row("clean", mean("", "flat"), [mean("", "g1"), mean("", "g2")])
    for s in (1, 2, 3, 4, 5):
        row(f"{s}", mean("", f"sev{s}_flat"), [mean("", f"sev{s}_g1"), mean("", f"sev{s}_g2")])
elif table == "table3":
    print("| backbone, rate | flat | late (each gamma) |")
    for b, rates in RATES3.items():
        for r, gs in rates.items():
            flat = mean(b, f"r{r}_flat")
            cells = [f"gamma={g:g}: {fmt(mean(b, f'r{r}_g{g:g}'), flat)}" for g in gs]
            print(f"| {b} r{r} | {fmt(flat)} | " + "; ".join(cells) + " |")
elif table == "table4":
    print("| backbone, rate | flat | late gamma=1 | late gamma=2 |")
    for b in ("DeiT-B", "ViT-S", "ViT-B"):
        for r in (8, 12):
            row(f"{b} r{r}", mean(b, f"r{r}_flat"), [mean(b, f"r{r}_g1"), mean(b, f"r{r}_g2")])
elif table == "table6":
    print("| backbone, rate | method | flat (mean +- std over seeds) | late (mean +- std) | delta |")
    for b, rates in RATES6.items():
        for r in rates:
            for method in METHODS6:
                per_seed = {k: [mean(b, f"{method}_r{r}_{k}_s{s}") for s in SEEDS] for k in ("flat", "late")}
                if any(v is None for vals in per_seed.values() for v in vals):
                    print(f"| {b} r{r} | {method} | incomplete | | |")
                    continue
                f, l = (st.mean(per_seed[k]) for k in ("flat", "late"))
                sf, sl = (st.pstdev(per_seed[k]) for k in ("flat", "late"))
                print(f"| {b} r{r} | {method} | {f:.2f} +- {sf:.2f} | {l:.2f} +- {sl:.2f} | {l - f:+.2f} |")
elif table == "table6_full":
    print("| backbone | method | full tokens (mean +- std over the seeds run) | seeds |")
    for b in RATES6:
        for method in METHODS6:
            v = [mean(b, f"{method}_full_s{s}") for s in SEEDS]
            v = [x for x in v if x is not None]
            print(f"| {b} | {method} | " + (f"{st.mean(v):.2f} +- {st.pstdev(v):.2f} | {len(v)} |" if v else "incomplete | 0 |"))
elif table == "uncompressed":
    print("| corruption | top-1, no reduction |")
    for j in json.load(open(out / "uncompressed_jobs.json")):  # the launcher's ImageNet-C order
        f = out / f"{j['tag']}_{j['dataset']}_{j['schedule']}.json"
        print(f"| {j['dataset'][len('inc_'):]} | {fmt(json.load(open(f))['top1'] if f.exists() else None)} |")
    print(f"| mean | {fmt(mean('', 'r0'))} |")
elif table == "nabirds":
    print("| NaBirds-C | flat | late gamma=2 |")
    for b in sorted({g for g, _ in runs}):
        row(b, mean(b, "flat"), [mean(b, "g2")])
elif table == "placement":
    print("| schedule | 15-corruption mean (flat and late are Table 1 runs) |")
    for k in ("early_r8", "early_r12"):
        print(f"| {k} | {fmt(mean('', k))} |")
elif table == "shapes":
    print("| late shape | 15-corruption mean (flat and the power law are Table 1 runs) |")
    for k in ("exp_b3", "step_w12"):
        print(f"| {k} | {fmt(mean('', k))} |")
elif table == "shiftsuites":
    print("| backbone, suite | flat | late gamma=2 |")
    for g in sorted({g for g, _ in runs}):
        row(g, mean(g, "flat"), [mean(g, "g2")])
else:
    sys.exit(f"unknown target {table}")
