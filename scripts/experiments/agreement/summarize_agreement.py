"""Aggregate the agreement runs of one model over their corruptions.

    python scripts/experiments/agreement/summarize_agreement.py [model_name] [out_dir]

Prints, per arm, the mean per-case KL(p_unreduced || p_reduced) over all corrupted cases and the share of
cases whose top-1 prediction differs from the unreduced model's, with the number of corruptions on which the
late arm has the lower share.
"""
import glob
import sys
from pathlib import Path

import numpy as np

model = sys.argv[1] if len(sys.argv) > 1 else "deit_small_patch16_224"
out = Path(sys.argv[2] if len(sys.argv) > 2 else "output/agreement")
files = sorted(glob.glob(str(out / f"{model}_*.npz")))
if not files:
    sys.exit(f"no {model}_*.npz under {out}")
arms = sorted({k.split("_", 1)[1].rsplit("_", 1)[0] for k in np.load(files[0]).files if k.startswith("corr_")})
kl = {a: [] for a in arms}
changed = {a: [] for a in arms}
per_corruption = {a: [] for a in arms}
for f in files:
    z = np.load(f)
    for a in arms:
        kl[a].append(z[f"corr_{a}_kl"])
        changed[a].append(~z[f"corr_{a}_same"].astype(bool))
        per_corruption[a].append(float(np.mean(~z[f"corr_{a}_same"].astype(bool))))
print(f"{model}: {len(files)} corruptions")
for a in arms:
    print(f"  {a:18s} mean KL {np.concatenate(kl[a]).mean():.3f}   top-1 changed {100 * np.concatenate(changed[a]).mean():.1f}%")
flat = next(a for a in arms if a.startswith("flat"))
late = next(a for a in arms if a.startswith("late"))
wins = sum(l < f for l, f in zip(per_corruption[late], per_corruption[flat]))
print(f"  late changes fewer predictions than flat on {wins}/{len(files)} corruptions")
