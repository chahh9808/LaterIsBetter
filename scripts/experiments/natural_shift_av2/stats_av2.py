"""Paired statistics for flat vs late on ImageNet-A / V2 / R / Sketch from per-image correctness dumps (the datasets whose runs exist)."""
import json, os, sys, csv, numpy as np
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3]); OUT = os.path.join(REPO, "output", "natural_shift_av2")
TAG = sys.argv[1] if len(sys.argv) > 1 else "vits_inx"; LABEL = sys.argv[2] if len(sys.argv) > 2 else TAG
from math import comb
def mcnemar_exact_p(b, c):  # two-sided exact binomial on discordant pairs
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
rng = np.random.default_rng(0)
rows = []; lines = [f"# ImageNet-A / V2 / R / Sketch, {LABEL}: flat r8 vs late (gamma=2, total_r=16), iso-GFLOPs, inference-only\n",
                    "seed 42, bs 64, full sets (A 7,500 / V2 10,000 / R 30,000 / Sketch 50,889). Paired statistics from per-image correctness of the SAME images under both schedules. McNemar = exact two-sided binomial test on discordant pairs; CI = 95% paired bootstrap (10,000 resamples of images).\n"]
lines.append("| dataset | N | full-token | flat | late | Δ late−flat | 95% CI | discordant (late✓flat✗ / late✗flat✓) | disagreement % | McNemar p | timm card full-token |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
CARD = {"imagenet-a": {"vit_small_patch16_224": 26.787, "vit_base_patch16_224": 49.733, "deit_small_patch16_224": 18.720, "deit_base_patch16_224": 27.173, "vit_base_patch16_clip_224.openai_ft_in1k": 47.453},
        "imagenet-v2": {"vit_small_patch16_224": 70.630, "vit_base_patch16_224": 75.320, "deit_small_patch16_224": 68.530, "deit_base_patch16_224": 70.970, "vit_base_patch16_clip_224.openai_ft_in1k": 76.090}}
for ds in ["imagenet-a", "imagenet-v2", "imagenet-r", "imagenet-sketch"]:
    if not os.path.exists(os.path.join(OUT, f"{TAG}_{ds}_flat.json")):
        continue
    m = {s: json.load(open(os.path.join(OUT, f"{TAG}_{ds}_{s}.json"))) for s in ["none", "flat", "late"]}
    fl = np.load(os.path.join(OUT, "preds", f"{TAG}_{ds}_flat.npy")); la = np.load(os.path.join(OUT, "preds", f"{TAG}_{ds}_late.npy"))
    assert fl.size == la.size == int(m["flat"]["num_samples"])
    assert abs(100 * fl.mean() - m["flat"]["top1"]) < 0.02 and abs(100 * la.mean() - m["late"]["top1"]) < 0.02, "dump/metrics mismatch"
    d = 100 * (la.mean() - fl.mean()); b = int((la & ~fl).sum()); c = int((~la & fl).sum())
    idx = rng.integers(0, fl.size, size=(10000, fl.size)); boots = 100 * (la[idx].mean(1) - fl[idx].mean(1)); lo, hi = np.percentile(boots, [2.5, 97.5])
    p = mcnemar_exact_p(b, c); model = m["flat"]["config"]["model"]; card = CARD.get(ds, {}).get(model)
    rows.append({"dataset": ds, "model": model, "N": fl.size, "full": m["none"]["top1"], "flat": m["flat"]["top1"], "late": m["late"]["top1"], "delta": d, "ci_lo": lo, "ci_hi": hi,
                 "late_only": b, "flat_only": c, "disagree_pct": 100 * (b + c) / fl.size, "mcnemar_p": p, "gflops_full": m["none"]["gflops"], "gflops_flat": m["flat"]["gflops"], "gflops_late": m["late"]["gflops"], "timm_card_full": card})
    lines.append(f"| {ds} | {fl.size} | {m['none']['top1']:.2f} | {m['flat']['top1']:.2f} | {m['late']['top1']:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {b} / {c} | {100*(b+c)/fl.size:.1f} | {p:.2g} | {card if card is not None else ''} |")
lines.append(f"\nGFLOPs: full {rows[0]['gflops_full']:.3f}, flat {rows[0]['gflops_flat']:.3f}, late {rows[0]['gflops_late']:.3f} (late <= flat: {rows[0]['gflops_late'] <= rows[0]['gflops_flat']}).")
md = "\n".join(lines); open(os.path.join(OUT, f"{TAG}_summary.md"), "w").write(md + "\n")
with open(os.path.join(OUT, f"{TAG}_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(md)
