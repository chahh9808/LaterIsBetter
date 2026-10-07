import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
"""Figure 2(b) for each of the five training-free reducers: 15-corruption-mean top-1 vs compute, each curve a
schedule swept over compression (DeiT-S, ImageNet-C severity 5). Values are the body Table 3 cells
(gamma=2 at r_base 4/6/8) and its gamma=1 appendix table (r_base 6/8/10); x is the flat schedule's GFLOPs
at each r_base, the budget the late schedule is calibrated to (late <= flat), as in make_fig1_calib_mean15; the uncompressed star of that figure is omitted.
-> output/figures/fig_calib_b_reducers.{pdf,png}"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5})
DEEP_RED, DEEP_NEWRED, GRAY = "#750014", "#AF021F", "#626a73"

ISO = json.load(open("configs/iso_gflops_gamma_table.json"))["table"]["S"]
GF = {r: ISO["_flat"][str(r)] for r in (4, 6, 8, 10)}       # flat GFLOPs per r_base (late plotted at flat x)

# Table 3 (gamma=2; r_base 4, 6, 8) and Table 15 (gamma=1; r_base 6, 8, 10), DeiT-S rows.
T = {  # reducer: {"flat": {r: acc}, "g2": {r: acc}, "g1": {r: acc}}
    "ToMe":   {"flat": {4: 38.83, 6: 38.44, 8: 37.99, 10: 37.41}, "g2": {4: 39.37, 6: 39.32, 8: 39.16}, "g1": {6: 39.13, 8: 38.89, 10: 38.58}},
    "EViT":   {"flat": {4: 37.50, 6: 36.69, 8: 35.98, 10: 35.16}, "g2": {4: 38.97, 6: 38.91, 8: 38.50}, "g1": {6: 38.10, 8: 37.87, 10: 37.62}},
    "ATS":    {"flat": {4: 34.95, 6: 34.50, 8: 33.92, 10: 33.35}, "g2": {4: 35.43, 6: 35.21, 8: 34.88}, "g1": {6: 34.99, 8: 34.61, 10: 34.38}},
    "ATC":    {"flat": {4: 38.28, 6: 37.96, 8: 37.55, 10: 37.04}, "g2": {4: 38.73, 6: 38.74, 8: 38.65}, "g1": {6: 38.53, 8: 38.35, 10: 38.26}},
    "PiToMe": {"flat": {4: 38.45, 6: 38.14, 8: 37.67, 10: 37.08}, "g2": {4: 38.99, 6: 38.67, 8: 37.90}, "g1": {6: 38.62, 8: 38.23, 10: 37.75}},
}
KIND = {"ToMe": "merge", "EViT": "prune", "ATS": "sample", "ATC": "cluster", "PiToMe": "merge"}

def curve(d): rs = sorted(d, reverse=True); return [GF[r] for r in rs], [d[r] for r in rs], rs

fig, axs = plt.subplots(2, 3, figsize=(15.0, 7.4), sharex=True)
for ax, (name, t) in zip(axs.ravel(), T.items()):
    x, y, _ = curve(t["flat"]); ax.plot(x, y, "o--", color=GRAY, lw=2.0, ms=6.5, label=f"flat (standard {name})", zorder=3)
    x, y, _ = curve(t["g1"]);   ax.plot(x, y, "s--", color=DEEP_NEWRED, lw=2.0, ms=6.5, label=r"late ($\gamma{=}1$)", zorder=4)
    x, y, rs = curve(t["g2"]);  ax.plot(x, y, "^-", color=DEEP_RED, lw=2.5, ms=7.5, label=r"late ($\gamma{=}2$, our default)", zorder=5)
    for xi, yi, r in zip(x, y, rs):
        off = (-4, -15) if (name, r) == ("PiToMe", 8) else (0, 8)   # PiToMe's r8 label would sit on the gamma=1 line
        ax.annotate(rf"$r_{{\mathrm{{base}}}}{{=}}{r}$", (xi, yi), textcoords="offset points", xytext=off, ha="center", fontsize=9, color=DEEP_RED)
    x, y, rs = curve(t["g1"])
    ax.annotate(rf"$r_{{\mathrm{{base}}}}{{=}}{rs[0]}$", (x[0], y[0]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=DEEP_NEWRED)
    # no uncompressed star here: it is a DeiT-S property already shown in Figure 2b, and at x=4.249 it would
    # read the budget axis as measured compute and flatten the ATS panel
    ax.set_title(f"{name} ({KIND[name]})"); ax.grid(alpha=0.3); ax.set_xlim(2.85, 3.85)
    ys = [v for k in ("flat", "g1", "g2") for v in t[k].values()]; ax.set_ylim(min(ys) - 0.4, max(ys) + 0.55)
for ax in axs[1]: ax.set_xlabel("GFLOPs per image")
for ax in axs[:, 0]: ax.set_ylabel("15-corruption mean top-1 (%)")
# sixth slot carries the legend (labels are shared; the flat entry is worded generically there)
h, l = axs[0, 0].get_legend_handles_labels(); l = ["flat (each method's default)" if x.startswith("flat") else x for x in l]
axs[1, 2].axis("off"); axs[1, 2].legend(h, l, loc="center", fontsize=13, framealpha=0.95)
axs[1, 1].set_xlabel("GFLOPs per image")
fig.tight_layout()
_os.makedirs("output/figures", exist_ok=True)
fig.savefig("output/figures/fig_calib_b_reducers.pdf", bbox_inches="tight")
fig.savefig("output/figures/fig_calib_b_reducers.png", dpi=150, bbox_inches="tight")
print("saved output/figures/fig_calib_b_reducers.{pdf,png}")
