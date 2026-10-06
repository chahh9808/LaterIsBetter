"""Table 4 and appendix rows for a backbone run by make_configs.py: ImageNet-C severity-5 mean
top-1 for flat / late gamma=1 / late gamma=2 at the main and the aggressive rate, plus the clean
ImageNet accuracies as a check against the published checkpoint.

    python scripts/experiments/clip_backbone/table4_clip.py [tag]
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CORRUPTIONS, REPO

tag = sys.argv[1] if len(sys.argv) > 1 else "clipb_openai"
out = REPO / "output" / "clip_backbone"
CELLS = {8: {"flat": "flat", "late_g1": "late_r8_g1", "late_g2": "late"},
         12: {"flat": "flat_r12", "late_g1": "late_r12_g1", "late_g2": "late_r12_g2"}}


def load(name):
    p = out / f"{tag}_{name}.json"
    return json.load(open(p)) if p.exists() else None


lines = [f"# {tag}: ImageNet-C severity 5, 15-corruption mean top-1 (%), iso-GFLOPs, inference-only",
         "| r_base | flat | late g=1 (delta) | late g=2 (delta) | GFLOPs flat / g1 / g2 |", "|---|---|---|---|---|"]
rows, per = [], {}
for r, cells in CELLS.items():
    acc, gf, missing = {}, {}, []
    for k, sfx in cells.items():
        vals = [load(f"inc_{c}_{sfx}") for c in CORRUPTIONS]
        if any(v is None for v in vals):
            missing.append(k)
            continue
        acc[k] = st.mean(v["top1"] for v in vals)
        gf[k] = st.mean(v["gflops"] for v in vals)
        per[(r, k)] = {c: v["top1"] for c, v in zip(CORRUPTIONS, vals)}
    if missing:
        lines.append(f"| {r} | missing: {missing} | | | |")
        continue
    d1, d2 = acc["late_g1"] - acc["flat"], acc["late_g2"] - acc["flat"]
    lines.append(f"| {r} | {acc['flat']:.2f} | {acc['late_g1']:.2f} ({d1:+.2f}) | {acc['late_g2']:.2f} ({d2:+.2f}) | "
                 f"{gf['flat']:.3f} / {gf['late_g1']:.3f} / {gf['late_g2']:.3f} |")
    pos = {k: sum(per[(r, k)][c] > per[(r, 'flat')][c] for c in CORRUPTIONS) for k in ("late_g1", "late_g2")}
    lines.append(f"|  | | positive {pos['late_g1']}/15 | positive {pos['late_g2']}/15 | |")
    rows.append({"r_base": r, **{k: acc[k] for k in acc}, "delta_g1": d1, "delta_g2": d2, **{f"gflops_{k}": gf[k] for k in gf}})
clean = {s: load(f"clean_{s}") for s in ("none", "flat", "late")}
if all(clean.values()):
    lines.append(f"\nclean ImageNet: no reduction {clean['none']['top1']:.2f}, flat {clean['flat']['top1']:.2f}, late {clean['late']['top1']:.2f}")
if per:
    lines += ["\n## per corruption", "| corruption | r8 flat | r8 g1 | r8 g2 | r12 flat | r12 g1 | r12 g2 |", "|---|---|---|---|---|---|---|"]
    cell = lambda r, k, c: f"{per[(r, k)][c]:.2f}" if (r, k) in per else ""
    for c in CORRUPTIONS:
        lines.append(f"| {c} | " + " | ".join(cell(r, k, c) for r in (8, 12) for k in ("flat", "late_g1", "late_g2")) + " |")
md = "\n".join(lines)
(out / f"{tag}_tab_arch.md").write_text(md + "\n")
if rows:
    with open(out / f"{tag}_tab_arch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(md)
