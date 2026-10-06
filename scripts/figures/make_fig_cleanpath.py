import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
"""Clean merge-path control: gaussian-noise top-1 against r_base for flat, the oracle that replays the clean
merge decisions, and late gamma=2, with the unreduced model as reference (DeiT-S, severity 5, iso-GFLOPs).
Values are the output of scripts/experiments/cleanpath/run_cleanpath.py. Writes output/figures/fig_cleanpath.{pdf,png}."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9.5})
DEEP_RED = "#750014"   # ours (late)
SILVER   = "#8b959e"   # flat baseline
DSILVER  = "#626a73"   # oracle control (clean decisions)
BLACK    = "#000000"

rb     = [4, 6, 8, 10]
flat   = [33.07, 32.35, 31.62, 30.73]   # own-path (standard ToMe)
oracle = [33.76, 33.45, 32.99, 32.43]   # forced clean merge decisions (oracle upper bound)
late   = [33.90, 33.84, 33.79, 33.44]   # late gamma=2 (its own corrupted decisions)
UNCOMP = 33.89                          # uncompressed model

fig, ax = plt.subplots(figsize=(4.7, 3.5))
ax.axhline(UNCOMP, ls=":", color=BLACK, alpha=0.5, lw=1.1, zorder=1, label="uncompressed")
ax.plot(rb, flat,   "o--", color=SILVER,   lw=2.0, ms=7, label="flat (standard ToMe)", zorder=3)
ax.plot(rb, oracle, "s--", color=DSILVER,  lw=2.0, ms=6, label="oracle clean decisions", zorder=4)
ax.plot(rb, late,   "^-",  color=DEEP_RED, lw=2.6, ms=8, label=r"late ($\gamma{=}2$, ours)", zorder=5)
ax.set_xlabel(r"compression  $r_{\mathrm{base}}$")
ax.set_ylabel("gaussian-noise top-1 (\%)".replace("\\%", "%"))
ax.set_xticks(rb)
ax.set_ylim(30, 34)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", framealpha=0.95)
fig.tight_layout()
import os
os.makedirs("output/figures", exist_ok=True)
fig.savefig("output/figures/fig_cleanpath.pdf")
fig.savefig("output/figures/fig_cleanpath.png", dpi=150)
print("saved output/figures/fig_cleanpath.pdf (+ .png)")
