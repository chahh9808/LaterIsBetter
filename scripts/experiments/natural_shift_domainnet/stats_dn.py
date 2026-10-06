"""Paired statistics for flat vs late on DomainNet-126 (source: real) from per-image correctness dumps."""
import json, os, sys, csv, numpy as np
from pathlib import Path
from math import comb
REPO = str(Path(__file__).resolve().parents[3]); OUT = os.path.join(REPO, "output", "natural_shift_domainnet")
TAG = sys.argv[1] if len(sys.argv) > 1 else "vits_dn"; LABEL = sys.argv[2] if len(sys.argv) > 2 else TAG
def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)
rng = np.random.default_rng(0); rows = []
lines = [f"# DomainNet-126, source = real, {LABEL}: flat r8 vs late (gamma=2, total_r=16), iso-GFLOPs, inference-only\n",
         "Eval preprocessing = source model's own (Resize 256, CenterCrop 224, mean/std 0.5). real = held-out source val (in-domain control); sketch/clipart/painting = full AdaContrast 126-class target lists. Paired statistics from per-image correctness of the same images under both schedules; McNemar = exact two-sided binomial test on discordant pairs; CI = 95% paired bootstrap (10,000 resamples).\n",
         "| domain | N | full-token | flat | late | Δ late−flat | 95% CI | discordant (late✓flat✗ / late✗flat✓) | disagreement % | McNemar p |", "|---|---|---|---|---|---|---|---|---|---|"]
for dom in ["real", "sketch", "clipart", "painting"]:
    m = {s: json.load(open(os.path.join(OUT, f"{TAG}_{dom}_{s}.json"))) for s in ["none", "flat", "late"]}
    fl = np.load(os.path.join(OUT, "preds", f"{TAG}_{dom}_flat.npy")); la = np.load(os.path.join(OUT, "preds", f"{TAG}_{dom}_late.npy"))
    assert fl.size == la.size == int(m["flat"]["num_samples"])
    assert abs(100 * fl.mean() - m["flat"]["top1"]) < 0.02 and abs(100 * la.mean() - m["late"]["top1"]) < 0.02, "dump/metrics mismatch"
    d = 100 * (la.mean() - fl.mean()); b = int((la & ~fl).sum()); c = int((~la & fl).sum())
    idx = rng.integers(0, fl.size, size=(10000, fl.size)); boots = 100 * (la[idx].mean(1) - fl[idx].mean(1)); lo, hi = np.percentile(boots, [2.5, 97.5]); p = mcnemar_exact_p(b, c)
    rows.append({"domain": dom, "model": m["flat"]["config"]["model"], "N": fl.size, "full": m["none"]["top1"], "flat": m["flat"]["top1"], "late": m["late"]["top1"], "delta": d, "ci_lo": lo, "ci_hi": hi,
                 "late_only": b, "flat_only": c, "disagree_pct": 100 * (b + c) / fl.size, "mcnemar_p": p, "gflops_full": m["none"]["gflops"], "gflops_flat": m["flat"]["gflops"], "gflops_late": m["late"]["gflops"]})
    lines.append(f"| {dom}{' (in-domain val)' if dom == 'real' else ''} | {fl.size} | {m['none']['top1']:.2f} | {m['flat']['top1']:.2f} | {m['late']['top1']:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {b} / {c} | {100*(b+c)/fl.size:.1f} | {p:.2g} |")
tg = [r for r in rows if r["domain"] != "real"]
lines.append(f"\nTarget-domain mean Δ (sketch/clipart/painting): {np.mean([r['delta'] for r in tg]):+.2f}pp; in-domain real val Δ {rows[0]['delta']:+.2f}pp.")
lines.append(f"GFLOPs: full {rows[0]['gflops_full']:.3f}, flat {rows[0]['gflops_flat']:.3f}, late {rows[0]['gflops_late']:.3f} (late <= flat: {rows[0]['gflops_late'] <= rows[0]['gflops_flat']}).")
md = "\n".join(lines); open(os.path.join(OUT, f"{TAG}_summary.md"), "w").write(md + "\n")
with open(os.path.join(OUT, f"{TAG}_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(md)
