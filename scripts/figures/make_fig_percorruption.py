import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
"""Per-corruption gain of the late schedule over flat at iso-GFLOPs, with the uncompressed model's gain behind it.
DeiT-S, r_base=8, gamma=2, ImageNet-C severity 5, 50k images per corruption. Each late bar is drawn in front of a
gray bar of the same width for the uncompressed model, so the gray left above a late bar is the accuracy late has not
recovered; labels are the late schedule's top-1. Corruptions are in the canonical ImageNet-C order, which is also
their group order. flat and late are the r_base=8 runs of Table 1; uncompressed is the same model with no reduction
(the `uncompressed` target of scripts/launch_table.py).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DR, SIL, BK, LSIL = "#750014", "#8b959e", "#000000", "#d5dade"
NAMES = ["gauss","shot","imp","defoc","glass","motn","zoom","snow","frost","fog","brt","cont","elas","pix","jpeg"]
FLAT = np.array([31.62,31.70,32.40,26.53,18.42,31.11,27.31,46.37,51.10,51.43,69.57,36.59,34.58,28.88,52.19])
LATE = np.array([33.79,33.58,34.85,27.57,18.72,32.24,28.75,47.62,52.45,52.97,70.18,37.31,34.79,29.74,52.82])
FULL = np.array([33.89,33.79,34.92,27.73,18.92,32.23,28.41,47.53,53.16,54.22,70.25,38.55,34.75,29.87,52.72])
GROUPS = [("noise", 0, 3), ("blur", 3, 7), ("weather", 7, 11), ("digital", 11, 15)]
D, G, x = LATE - FLAT, FULL - FLAT, np.arange(len(NAMES))
WIDTH, YMAX, HEADER_DROP, FIG_H = 0.58, 3.45, 0.12, 1.468


def render(out):
    """Sized for a 0.45\\linewidth minipage: figsize is the final size, so point sizes render 1:1."""
    plt.rcParams.update({"font.size":6.0,"axes.labelsize":6.3,"xtick.labelsize":5.4,"ytick.labelsize":5.6})
    fig, ax = plt.subplots(figsize=(2.20, FIG_H))
    for _, a, _b in GROUPS[1:]:
        ax.axvline(a - 0.5, color=BK, lw=0.5, alpha=0.30, zorder=1)
    ax.bar(x, G, width=WIDTH, color=LSIL, edgecolor="none", zorder=2)
    ax.bar(x, D, width=WIDTH, color=DR, edgecolor="none", zorder=3)
    ax.axhline(0, color=SIL, lw=1.4, zorder=4)
    for xi, b in zip(x, LATE):
        ax.annotate(f"{b:.1f}", (xi, D[xi]), textcoords="offset points", xytext=(0, 2.5), ha="center",
                    va="bottom", rotation=90, fontsize=4.3, color=DR, zorder=9)
    for g, a, b in GROUPS:
        ax.text((a + b - 1) / 2, YMAX - HEADER_DROP, g, ha="center", va="top", fontsize=4.9, color=BK, alpha=0.75)
    ax.legend([Patch(color=DR), Patch(color=LSIL)], ["late (ours)", "uncompressed"], fontsize=4.2, loc="upper right",
              bbox_to_anchor=(1.0, (YMAX - HEADER_DROP - 0.21) / (YMAX + 0.12)), frameon=True, framealpha=0.92,
              edgecolor="none", handlelength=1.0, handletextpad=0.3, borderpad=0.2, borderaxespad=0.25,
              labelspacing=0.15)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=90)
    ax.set_ylabel("gain over flat (pp)", labelpad=1.5)
    ax.set_xlim(-0.7, len(NAMES) - 0.3); ax.set_ylim(-0.12, YMAX)
    ax.set_yticks([0, 1, 2, 3]); ax.grid(alpha=0.25, axis="y", lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.4)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} | all 15 improve over flat: {bool((D > 0).all())}")
if __name__ == "__main__":
    _os.makedirs("output/figures", exist_ok=True)
    render("output/figures/fig_percorruption_wrap.pdf")
