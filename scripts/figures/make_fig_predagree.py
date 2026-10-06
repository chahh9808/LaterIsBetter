import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
"""Per-image agreement with the unreduced model, a 2x2 of (input: clean / corrupted) x (schedule: flat / late).
Each case is one image: x = the probability the unreduced model assigns to its own predicted class c*, y = the
probability the reduced model assigns to that class; exact agreement is the dashed diagonal. Colour is the local
density of cases on a scale shared by the four panels. Reads output/agreement/<model>_<corruption>.npz from
scripts/experiments/agreement/run_agreement.py and writes output/figures/fig_predagree.{pdf,png}.

    python scripts/figures/make_fig_predagree.py [corruption] [model_name]
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

CORR = sys.argv[1] if len(sys.argv) > 1 else "gaussian_noise"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deit_small_patch16_224"
BRIGHT_RED = "#ff1423"
NBINS, MARKER, VMAX_PCT, CMAP = 200, 3.0, 90.0, "magma_r"

z = np.load(f"output/agreement/{MODEL}_{CORR}.npz")
arms = sorted({k.split("_", 1)[1].rsplit("_", 1)[0] for k in z.files if k.startswith("clean_")})
FLAT = next(a for a in arms if a.startswith("flat")); LATE = next(a for a in arms if a.startswith("late"))
rows = [("Flat", FLAT), ("Late", LATE)]
cols = [("Clean", "clean"), ("Noise" if "noise" in CORR else "Corrupted", "corr")]

# local density of every case, and one colour ceiling for all four panels
dens = {}
for _, arm in rows:
    for _, cond in cols:
        X, Y = z[f"{cond}_{arm}_x"], z[f"{cond}_{arm}_y"]
        H, xe, ye = np.histogram2d(X, Y, bins=NBINS, range=[[0, 1], [0, 1]])
        bx = np.clip(np.digitize(X, xe) - 1, 0, NBINS - 1); by = np.clip(np.digitize(Y, ye) - 1, 0, NBINS - 1)
        dens[(arm, cond)] = H[bx, by]
pooled = np.concatenate(list(dens.values()))
vmax = max(float(np.percentile(pooled, VMAX_PCT)), 2.0)

fig, axarr = plt.subplots(2, 2, figsize=(9.6, 9.4))
for ri, (rlab, arm) in enumerate(rows):
    for ci, (clab, cond) in enumerate(cols):
        ax = axarr[ri][ci]
        X, Y, d = z[f"{cond}_{arm}_x"], z[f"{cond}_{arm}_y"], dens[(arm, cond)]
        order = np.argsort(d)  # dense cases drawn last, on top
        sc = ax.scatter(X[order], Y[order], c=d[order], s=MARKER, marker="o", linewidths=0, cmap=CMAP,
                        rasterized=True, norm=LogNorm(vmin=1, vmax=vmax))
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=BRIGHT_RED)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"({clab}, {rlab})", fontsize=13, pad=6)
        ax.text(0.04, 0.96, f"KL {z[f'{cond}_{arm}_kl'].mean():.3f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.6", alpha=0.88))
        if ci == 0:
            ax.set_ylabel(r"$p_{\mathrm{reduced}}(c^{\star})$", fontsize=14)
        if ri == 1:
            ax.set_xlabel(r"$p_{\mathrm{unreduced}}(c^{\star})$", fontsize=14)
fig.subplots_adjust(left=0.11, right=0.86, bottom=0.08, top=0.93, hspace=0.20, wspace=0.16)
# the panels keep a square aspect, so read their drawn positions and fit the colourbar to them
fig.canvas.draw()
ptop, pbot = axarr[0][1].get_position(), axarr[1][1].get_position()
cax = fig.add_axes([max(ptop.x1, pbot.x1) + 0.020, pbot.y0, 0.018, ptop.y1 - pbot.y0])
cb = fig.colorbar(sc, cax=cax, extend="max")
cb.set_label("local density (cases per bin)", fontsize=13, labelpad=2)
cb.ax.tick_params(labelsize=10)
_os.makedirs("output/figures", exist_ok=True)
fig.savefig("output/figures/fig_predagree.pdf", bbox_inches="tight")
fig.savefig("output/figures/fig_predagree.png", dpi=190, bbox_inches="tight")
print("wrote output/figures/fig_predagree.pdf")
