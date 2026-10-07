import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Large source fonts: the figure is scaled to the column width.
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10.5,
})

# Deep Red marks the main setting (gamma=2 in (a),(b), r_base=8 in (c)); Silver is the flat baseline.
DEEP_RED   = "#750014"   # headline setting: gamma=2 in (a),(b); r_base=8 in (c)
SILVER     = "#8b959e"   # flat / const baseline: panels (a),(b)
BLACK      = "#000000"   # axes / uncompressed reference / optimum markers
DSILVER    = "#626a73"   # dark silver gray: panel (a) gamma=1.5
DEEP_NEWRED   = "#AF021F"   # headline setting: gamma=2 in (a),(b); r_base=8 in (c)
DGRAY      = "#40464c"   # dark gray: panel (a) gamma=2.0
# Panel (c): red intensity tracks compression (darker = more aggressive r_base), r8 the mid reference.
RED_R4     = "#ef9a8e"   # panel (c): r_base=4  (lightest red, least compression)
RED_R8     = "#b21e34"   # panel (c): r_base=8  (mid deep red, reference)
RED_R12    = "#5e0010"   # panel (c): r_base=12 (darkest red, most compression)
MAIN = DEEP_RED          # alias
GRAY = DSILVER            # alias (flat/const baseline color)

# ---------------------------------------------------------------
# Panel (a): schedule shape r_l vs block, gamma in {0,1,1.5,2}, r_base=8 (default), L=12
#   r_l = (r_base*L) * (l/L)^g / sum_l (l/L)^g   (matches build_late_concentrated_schedule)
# ---------------------------------------------------------------
L = 12; RBASE = 8
blocks = list(range(1, L+1))
def schedule(g):
    w = [(l/L)**g for l in blocks]
    s = sum(w)
    return [RBASE*L * wi / s for wi in w]
gammas_a = [(0.0, GRAY,    "o", "--",  r"$\gamma{=}0$ (flat)"),
            (1.0, DEEP_NEWRED, "s", "--", r"$\gamma{=}1$"),
            (2.0, MAIN,    "^", "-",  r"$\gamma{=}2$ (our default)")]

# Panel (b): iso-GFLOPs frontier, ImageNet-C 15-corruption mean at severity 5, DeiT-S: late gamma=1 and
# gamma=2 against flat at r_base 2..12 (Table 1 runs; late is plotted at the flat GFLOPs it is calibrated to).
levels  = [2,    4,    6,    8,    10,   12]
gflops  = [3.986,3.724,3.462,3.199,2.936,2.674]   # flat GFLOPs per r_base (late sits <= these; plotted at flat x)
flat    = [39.14,38.83,38.44,37.99,37.41,36.67]
late_g1 = [39.34,39.21,39.13,38.89,38.58,37.88]   # tr 3/6/9/12/15/18
late_g2 = [39.40,39.37,39.32,39.16,38.66,37.77]   # tr 4/8/13/16/21/27
uncomp  = (4.249, 39.40); _BLAB="15-corruption mean top-1 (%)"; _YTOP=39.8
ann = {2:(0,8,"center"), 4:(0,8,"center"), 6:(0,8,"center"), 8:(0,10,"center"), 10:(-12,10,"center"), 12:(12,-15,"center")}

# Panel (c): 15-corruption mean accuracy against gamma at r_base 4, 8, 12 (Table 1 runs); gamma=0 is flat.
ucurve = {  # r_base: (gammas, 15-corr mean acc, color, marker)
    4:  ([0,0.5,1.0,1.5,2.0,2.5,3.0],         [38.83,39.06,39.21,39.33,39.37,39.43,39.44], RED_R4,  "o"),
    8:  ([0,0.5,1.0,1.5,2.0,2.5,3.0],         [37.99,38.56,38.89,39.10,39.16,39.13,39.15], RED_R8,  "^"),
    12: ([0,0.5,1.0,1.5,2.0,2.5,3.0],         [36.67,37.54,37.88,37.90,37.77,37.53,37.54], RED_R12, "s"),
}

def draw_a(ax):
    for g, c, mk, ls, lab in gammas_a:
        ax.plot(blocks, schedule(g), marker=mk, ls=ls, color=c, lw=2.2, ms=6, label=lab,
                zorder=4 if g==2.0 else 3)
    ax.set_xlabel(r"layer index $\ell$"); ax.set_ylabel(r"tokens removed $r_\ell$")
    ax.set_title(r"(a) schedule shape")
    ax.set_xticks([1,3,6,9,12]); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

def draw_b(ax):
    ax.plot(gflops, flat, "o--", color=GRAY, lw=2.0, ms=7, label="flat (standard ToMe)", zorder=3)
    ax.plot(gflops, late_g1, "s--", color=DEEP_NEWRED, lw=2.0, ms=7,
            label=r"late ($\gamma{=}1$)", zorder=4)
    ax.plot(gflops, late_g2, "^-", color=MAIN, lw=2.6, ms=8,
            label=r"late ($\gamma{=}2$, our default)", zorder=5)
    ax.scatter([uncomp[0]], [uncomp[1]], marker="*", s=240, color="black", zorder=6, label="uncompressed")
    ax.axhline(uncomp[1], ls=":", color="black", alpha=0.35, lw=1, zorder=1)
    for g, y, r in zip(gflops, late_g2, levels):
        dx, dy, ha = ann[r]
        ax.annotate(rf"$r_{{\mathrm{{base}}}}{{=}}{r}$", (g, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=9, color=MAIN)
    ax.set_xlabel("GFLOPs per image"); ax.set_ylabel(_BLAB)
    ax.set_title("(b) accuracy vs compute")
    ax.set_xlim(left=2.6); ax.set_ylim(top=_YTOP); ax.legend(loc="lower right", framealpha=0.95); ax.grid(alpha=0.3)

def draw_c(ax, delta=False):
    for r in (4, 8, 12):
        gs, acc, c, mk = ucurve[r]
        base = acc[0]                       # gamma=0 (flat) reference for Delta
        gsp, accp = gs[1:], acc[1:]         # drop the gamma=0 point from the plot (x starts at 0.5)
        y = [a - base for a in accp] if delta else accp
        ax.plot(gsp, y, marker=mk, ls="-", color=c, lw=2.4, ms=6, zorder=3,
                label=rf"$r_{{\mathrm{{base}}}}{{=}}{r}$")
        # No optimum marker in (c): each curve's peak is already visible, and the star is
        # reserved for the uncompressed reference in (b), so one glyph keeps one meaning.
    if delta:
        ax.axhline(0.0, ls="-", color=BLACK, alpha=0.45, lw=1, zorder=1)   # flat baseline (Delta=0)
        ax.annotate(r"$\gamma{=}2$", (2.0, 0.04), fontsize=11, rotation=90,
                    va="bottom", ha="right", color="black", alpha=0.75)
        ax.set_ylabel(r"$\Delta$ 15-corr mean (pp)")
        ax.set_title(r"(c) $\gamma$ sensitivity")
        ax.legend(loc="upper left")
    else:
        ax.annotate(r"$\gamma{=}2$", (2.0, 36.78), fontsize=11, rotation=90,
                    va="bottom", ha="right", color="black", alpha=0.75)
        ax.set_ylabel("15-corruption mean top-1 (%)")
        ax.set_title(r"(c) $\gamma$ sensitivity")
        ax.legend(loc="lower right")
    ax.axvline(2.0, ls=":", color="black", alpha=0.4, lw=1)
    ax.set_xlim(left=0.4); ax.set_xticks([0.5,1.0,1.5,2.0,2.5,3.0])
    ax.set_xlabel(r"schedule exponent $\gamma$"); ax.grid(alpha=0.3)

import os
os.makedirs("output/figures", exist_ok=True)   # paper figures live in output/figures/

def build_3panel(path, delta):
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.4))
    draw_a(axs[0]); draw_b(axs[1]); draw_c(axs[2], delta=delta)
    fig.tight_layout(); fig.savefig(path); fig.savefig(path.replace(".pdf",".png"),dpi=150); plt.close(fig)

build_3panel("output/figures/fig_calib_3panel.pdf",     delta=True)
build_3panel("output/figures/fig_calib_3panel_abs.pdf", delta=False)
print("saved output/figures/fig_calib_3panel.pdf (Delta, 15-corr) + fig_calib_3panel_abs.pdf (absolute)")
