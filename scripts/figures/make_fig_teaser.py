import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
"""Merged teaser: (a,b) the problem and what the schedule recovers, (c) what the schedule is.
The scatter-panel variant of (c) lives in make_fig_teaser_scatter.py.
Panel order follows the introduction, which states the problem before the remedy. (a,b) reuse the
motivation bars of make_fig_compdrop_bars.py; (c) draws the realized DeiT-S schedules of fig:placement.
Everything is gaussian noise sev5 / DeiT-S, so the panels share one scale. -> output/figures/fig_teaser.{pdf,png}"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DR, GREY, BK = "#750014", "#636b73", "#000000"
GREY_FILL = "#9aa3ab"   # flat fills; GREY stays for flat text, which needs the darker tone on white
BACK = "#dfe3e6"
INIT = 196
FLAT_T = [INIT,188,180,172,164,156,148,140,132,124,116,108,100]
LATE_T = [INIT,196,195,192,187,180,169,155,136,112,82,46,23]
G_FLAT, G_LATE, G_FULL = 3.199, 3.147, 4.25
A_FULL, A_FLAT, A_LATE = 33.89, 31.62, 33.79          # gaussian noise sev5, r_base=8
D_ = "output/teaser"  # written by: python scripts/launch_table.py teaser, then scripts/experiments/sweep.py
_jobs = json.load(open(f"{D_}/teaser_jobs.json"))
_files = {f"{j['schedule']}_{j['dataset']}": f"{D_}/{j['tag']}_{j['dataset']}_{j['schedule']}.json" for j in _jobs}
RBS = [0, 2, 4, 6, 8, 10]
acc = lambda tag: json.load(open(_files[tag]))["top1"]
d = np.arange(13)
DX = 13.2

plt.rcParams.update({"font.size":10.5})
fig = plt.figure(figsize=(10.0, 6.25))
outer = fig.add_gridspec(2, 1, height_ratios=[3.30, 2.55], hspace=0.53,
                         left=0.062, right=0.985, top=0.945, bottom=0.075)

# ---------------- (a,b) the problem, across compression -------------------------------
top = outer[0].subgridspec(1, 4, width_ratios=[0.196, 1.0, 1.0, 0.064], wspace=0.46)
for i, (title, split, ylim, yticks) in enumerate(
        [("(a) in-distribution (clean)", "clean", (60, 86), [60, 65, 70, 75, 80]),
         ("(b) out-of-distribution (gauss. noise, sev 5)", "gaussian_noise", (20, 46), [20, 25, 30, 35, 40])]):
    ax = fig.add_subplot(top[0, i + 1])
    flat = np.array([acc(f"r{rb}_{split}") for rb in RBS])
    late = np.array([acc(f"r0_{split}") if rb == 0 else acc(f"late_r{rb}_{split}") for rb in RBS])
    x = np.arange(len(RBS))
    ax.bar(x, flat, 0.66, color=GREY_FILL, zorder=3, label="default (flat schedule)")
    ax.bar(x, late, 0.66, color=DR, zorder=2, label="ours (late-concentrated)")
    ax.axhline(flat[0], color=BK, lw=1.0, ls=(0, (4, 3)), alpha=0.75, zorder=4,
               label="uncompressed ($r_{\mathrm{base}}{=}0$)")
    for xi, (f, l) in enumerate(zip(flat, late)):
        ax.annotate(f"{f:.1f}", (xi, f), textcoords="offset points", xytext=(0, -5), ha="center",
                    va="top", fontsize=9.5, color="black")
        if l - f > 0.4:
            ax.annotate(f"+{l-f:.1f}", (xi, max(l, f)), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=10.5, color=DR, fontweight="bold")
    ax.set_title(title, fontsize=11.5, pad=5)
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RBS])
    ax.set_xlabel(r"compression  $r_{\mathrm{base}}$", labelpad=2)
    ax.set_ylim(*ylim); ax.set_yticks(yticks); ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_ylabel("top-1 accuracy (%)", labelpad=2)
    if i == 1:
        ax.legend(frameon=True, loc="upper left", fontsize=9.4)

# ---------------- (c) what the schedule is --------------------------------------------
bot = outer[1].subgridspec(2, 3, width_ratios=[0.95, 4.98, 0.60], hspace=0.34, wspace=0.115)
XL0, XL1 = -3.95, 16.4   # ribbon x-range; the right margin holds the "N tokens" label and its token row
axin = fig.add_subplot(bot[:, 0].subgridspec(3, 1, height_ratios=[0.26, 1.0, 0.26])[1, 0])
axr = [fig.add_subplot(bot[0, 1]), fig.add_subplot(bot[1, 1])]
axg = [fig.add_subplot(bot[0, 2]), fig.add_subplot(bot[1, 2])]
import matplotlib.image as mpimg
axin.imshow(mpimg.imread(str(Path(__file__).resolve().parent / 'assets' / 'mergevis_input.png'))); axin.set_axis_off()
# 14x14 patch grid over the image content (border of the asset sits at px 4-5 / 558-559),
# so the 196 tokens the label names are visible as the patches they actually are
_c0, _c1, _NP = 5.5, 557.5, 14
for _i in range(1, _NP):
    _q = _c0 + _i * (_c1 - _c0) / _NP
    axin.plot([_q, _q], [_c0, _c1], color='white', lw=0.35, alpha=0.5, zorder=3)
    axin.plot([_c0, _c1], [_q, _q], color='white', lw=0.35, alpha=0.5, zorder=3)
axin.set_title(f'input, $\\bf{{{INIT}}}$ tokens\n(gauss. noise, sev 5)', fontsize=10.0, pad=3)

tok_labels = []
for ax, y, col, name, g in ((axr[0], FLAT_T, GREY_FILL, "default (flat)", G_FLAT),
                            (axr[1], LATE_T, DR, "ours (late)", G_LATE)):
    y = np.array(y, float); tc = GREY if col == GREY_FILL else col
    ax.fill_between([0, 12], -INIT/2, INIT/2, color=BACK, lw=0, zorder=1)
    ax.fill_between(d, -y/2, y/2, color=col, alpha=0.92, lw=0, zorder=3)
    for k in range(13):
        ax.plot([k, k], [-y[k]/2, y[k]/2], color="white", lw=0.7, alpha=0.55, zorder=4)
    ax.text(-0.45, INIT/2 * 0.16, name, ha="right", va="bottom", fontsize=11.3, color=tc,
            fontweight="bold")
    ax.text(-0.45, -INIT/2 * 0.16, f"{g:.2f} GFLOPs", ha="right", va="top", fontsize=8.8,
            color="#6f767d")
    tok_labels.append(ax.text(12.22, 0, f"$\\bf{{{y[-1]:.0f}}}$ tokens", ha="left", va="center",
                              fontsize=11.5, color=tc))
    ax.set_xlim(XL0, XL1); ax.set_ylim(-INIT/2 * 1.16, INIT/2 * 1.16); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
axr[1].sharex(axr[0]); axr[0].tick_params(labelbottom=False)
axr[1].set_xticks(range(0, 13, 2)); axr[1].set_xlabel("layer", labelpad=2)
axr[1].xaxis.set_label_coords((6 - XL0) / (XL1 - XL0), -0.27)

for ax, a, col in ((axg[0], A_FLAT, GREY_FILL), (axg[1], A_LATE, DR)):
    gap = A_FULL - a; tc = GREY if col == GREY_FILL else col
    ax.barh([0], [gap], height=0.5, color=col, zorder=3)
    ax.text(gap + 0.09, -0.10 if col == DR else 0.0, f"\u2212{gap:.2f}", ha="right",
            va="center", fontsize=11.5, color=tc, fontweight="bold")
    if col == DR:
        # the span from where flat's shortfall ends to ours: the gain at this operating point
        ax.annotate("", xy=(gap, 0.14), xytext=(A_FULL - A_FLAT, 0.14),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6, mutation_scale=17,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlim(0, 3.2); ax.set_ylim(-0.62, 0.62); ax.set_yticks([]); ax.invert_xaxis()
    ax.set_xticks([0, 1, 2]); ax.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6); ax.tick_params(length=2, pad=1.5, labelsize=7.5)
axg[0].tick_params(labelbottom=False)
axg[1].set_xlabel("gap to\nuncompressed (pp)", fontsize=8.2, labelpad=2, linespacing=1.15)

fig.canvas.draw()
# centre the heading on the whole (c) row (ribbons plus gap bars), not on the ribbon alone
XC = (axin.get_position().x0 + axg[1].get_position().x1) / 2 + 0.010
_y = fig.transFigure.inverted().transform(axr[0].transAxes.transform((0, 1)))[1]
fig.text(XC, _y + 0.076, "(c) same compute, different schedule: where the token budget is spent over depth",
         ha="center", va="top", fontsize=11.5)
_sw = fig.add_axes([XC - 0.164, _y + 0.024, 0.016, 0.020]); _sw.set_xticks([]); _sw.set_yticks([])
_sw.set_facecolor(BACK)
for _s in _sw.spines.values():
    _s.set_color("#b9c0c6"); _s.set_linewidth(0.7)
fig.text(XC - 0.142, _y + 0.034, f"uncompressed ({A_FULL:.2f}% top-1, {G_FULL:.2f} GFLOPs)",
         ha="left", va="center", fontsize=9.8, color="#2b2f33")

# token rows: a few model tokens under each "N tokens" label, each the size of one input patch,
# an ellipsis standing in for the rest; the flat row runs twice as far right as the late one
_r = fig.canvas.get_renderer(); _f = fig.transFigure.inverted()
_p0 = _f.transform(axin.transData.transform((_c0, _c0)))
_p1 = _f.transform(axin.transData.transform((_c0 + (_c1 - _c0) / _NP, _c0 + (_c1 - _c0) / _NP)))
TOKEN_SCALE = 2.0   # model-token size relative to one input patch
_pw, _ph = TOKEN_SCALE * abs(_p1[0] - _p0[0]), TOKEN_SCALE * abs(_p1[1] - _p0[1])   # figure fraction
for _t, _pat, _fill, _tc in ((tok_labels[0], "ssss.sss", GREY_FILL, GREY),
                             (tok_labels[1], "ss.s", DR, DR)):
    _bb = _f.transform(_t.get_window_extent(_r).get_points())
    _ax = fig.add_axes([_bb[0][0], _bb[0][1] - 1.45 * _ph, len(_pat) * _pw, _ph])
    _ax.set_axis_off(); _ax.set_xlim(0, len(_pat)); _ax.set_ylim(0, 1)
    for _i, _ch in enumerate(_pat):
        if _ch == "s":
            _ax.add_patch(plt.Rectangle((_i + 0.08, 0.08), 0.84, 0.84, color=_fill, lw=0))
        else:   # ellipsis as three dots drawn to scale, so it survives any TOKEN_SCALE
            for _k in (-0.28, 0.0, 0.28):
                _ax.add_patch(plt.Circle((_i + 0.5 + _k, 0.5), 0.075, color=_tc, lw=0))

for ext in ("pdf", "png"):
    # the gap-axis label reaches past the canvas edge; make sure the tight bbox keeps all of it
    fig.savefig(f"output/figures/fig_teaser.{ext}", dpi=150, bbox_inches="tight",
                bbox_extra_artists=[axg[1].xaxis.label])
print("saved output/figures/fig_teaser.{pdf,png}")
