"""Cross-backbone table for ImageNet-A / V2 from the per-tag summary CSVs written by stats_av2.py."""
import csv, os, glob
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[3]); OUT = os.path.join(REPO, "output", "natural_shift_av2")
LABEL = {"vits_inx": "ViT-S", "vitb_inx": "ViT-B", "deits_inx": "DeiT-S", "deitb_inx": "DeiT-B", "clipb_inx": "CLIP ViT-B"}
rows = []
for tag in ["vits_inx", "vitb_inx", "deits_inx", "deitb_inx", "clipb_inx"]:
    p = os.path.join(OUT, f"{tag}_summary.csv")
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)): r["backbone"] = LABEL[tag]; rows.append(r)
lines = ["# ImageNet-A / ImageNet-V2 across backbones: flat r8 vs late (gamma=2, total_r=16), iso-GFLOPs, inference-only, seed 42\n",
         "Paired per-image statistics (95% bootstrap CI, exact McNemar). full = no reduction; timm card = published full-token accuracy for the same weights.\n"]
for ds in ["imagenet-a", "imagenet-v2"]:
    sub = [r for r in rows if r["dataset"] == ds]
    lines += [f"## {ds}", "| backbone | N | full | timm card | flat | late | Δ late−flat | 95% CI | McNemar p |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sub:
        card = r["timm_card_full"] if r["timm_card_full"] not in ("", "None") else ""
        lines.append(f"| {r['backbone']} | {r['N']} | {float(r['full']):.2f} | {card} | {float(r['flat']):.2f} | {float(r['late']):.2f} | {float(r['delta']):+.2f} | [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}] | {float(r['mcnemar_p']):.2g} |")
    d = [float(r["delta"]) for r in sub]; lines.append(f"\nmean Δ {sum(d)/len(d):+.2f}pp, positive {sum(x>0 for x in d)}/{len(d)}, CI excludes 0 in {sum(float(r['ci_lo'])>0 for r in sub)}/{len(sub)}\n")
md = "\n".join(lines); open(os.path.join(OUT, "all_backbones_summary.md"), "w").write(md + "\n"); print(md)
