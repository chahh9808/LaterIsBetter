"""Cross-backbone DomainNet-126 table from per-tag summary CSVs written by stats_dn.py."""
import csv, os
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3]); OUT = os.path.join(REPO, "output", "natural_shift_domainnet")
LABEL = {"vits_dn": "ViT-S", "vitb_dn": "ViT-B", "deits_dn": "DeiT-S", "deitb_dn": "DeiT-B", "clipb_dn": "CLIP ViT-B"}
rows = []
for tag in ["vits_dn", "vitb_dn", "deits_dn", "deitb_dn", "clipb_dn"]:
    p = os.path.join(OUT, f"{tag}_summary.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)): r["backbone"] = LABEL[tag]; rows.append(r)
lines = ["# DomainNet-126 (source real) across backbones: flat r8 vs late (gamma=2, total_r=16), iso-GFLOPs, inference-only\n",
         "Each backbone is its own source model, finetuned on the real domain with one recipe. Paired per-image statistics (95% bootstrap CI, exact McNemar).\n"]
for dom in ["real", "sketch", "clipart", "painting"]:
    sub = [r for r in rows if r["domain"] == dom]
    lines += [f"## {dom}{' (in-domain val)' if dom == 'real' else ''}", "| backbone | N | full | flat | late | Δ late−flat | 95% CI | McNemar p |", "|---|---|---|---|---|---|---|---|"]
    for r in sub: lines.append(f"| {r['backbone']} | {r['N']} | {float(r['full']):.2f} | {float(r['flat']):.2f} | {float(r['late']):.2f} | {float(r['delta']):+.2f} | [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}] | {float(r['mcnemar_p']):.2g} |")
    if sub:
        d = [float(r["delta"]) for r in sub]; lines.append(f"\nmean Δ {sum(d)/len(d):+.2f}pp, positive {sum(x>0 for x in d)}/{len(d)}, CI excludes 0 in {sum(float(r['ci_lo'])>0 for r in sub)}/{len(sub)}\n")
md = "\n".join(lines); open(os.path.join(OUT, "all_backbones_summary.md"), "w").write(md + "\n"); print(md)
