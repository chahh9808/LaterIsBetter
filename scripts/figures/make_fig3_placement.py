import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # cwd -> repo root
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
DEEP_RED="#750014"; SILVER="#8b959e"; DGRAY="#40464c"
INIT=196
# realized tokens-remaining after each block (depth 0 = before block 1 = INIT), DeiT-S, gamma=2 late
# r_base=8
r8_early=[INIT,183,172,163,155,149,144,141,139,137,136,136,136]
r8_flat =[INIT,188,180,172,164,156,148,140,132,124,116,108,100]
r8_late =[INIT,196,195,192,187,180,169,155,136,112,82,46,23]
g8_early,g8_flat,g8_late=3.290,3.199,3.147
# r_base=12
r12_early=[INIT,175,157,142,130,120,113,108,104,102,101,100,100]
r12_flat =[INIT,184,172,160,148,136,124,112,100,88,76,64,52]
r12_late =[INIT,195,193,188,180,168,150,126,94,54,27,14,7]
g12_early,g12_flat,g12_late=2.710,2.675,2.666
# accuracy bars, 15-corruption mean top-1: early from the placement runs, flat and late from Table 1
early=[37.12,34.44]; flat=[37.99,36.67]; late=[39.16,37.77]
dearly=[-0.87,-2.23]; dlate=[+1.17,+1.10]
depth=np.arange(0,13)


# ---------------- main figure: r_base=8 curves + accuracy bars (compact, for a wrapfigure)
fig=plt.figure(figsize=(2.95,1.45))
_gs=fig.add_gridspec(1,2,width_ratios=[1,1.22],wspace=0.42)
axc=fig.add_subplot(_gs[0,0]); axb=fig.add_subplot(_gs[0,1])

def curve_panel(ax,early_y,flat_y,late_y,title):
    ax.plot(depth,early_y,color=DGRAY,  lw=1.2,marker="o",ms=1.8,zorder=3,label=f"early {g8_early:.2f}G")
    ax.plot(depth,flat_y, color=SILVER, lw=1.2,marker="s",ms=1.8,zorder=3,label=f"flat {g8_flat:.2f}G")
    ax.plot(depth,late_y, color=DEEP_RED,lw=1.8,marker="o",ms=2.1,zorder=4,label=f"late $\\gamma$2 {g8_late:.2f}G")
    ax.set_title(title,fontsize=7.0,pad=2)
    ax.set_xlabel(r"layer depth $\ell$",fontsize=6.0,labelpad=1)
    ax.set_xticks([0,4,8,12]); ax.set_ylim(0,205)
    ax.tick_params(labelsize=5.4,pad=1); ax.grid(alpha=0.3,zorder=0)
    ax.legend(fontsize=4.6,loc="lower left",handlelength=0.9,handletextpad=0.3,borderpad=0.25,
              labelspacing=0.18,borderaxespad=0.25,frameon=True,framealpha=0.9)

curve_panel(axc,r8_early,r8_flat,r8_late,r"(a) $r_{\mathrm{base}}{=}8$")
axc.set_ylabel("tokens left",fontsize=6.0,labelpad=1)

x=np.arange(2); w=0.26
groups=[r"$r_{\mathrm{base}}{=}8$",r"$r_{\mathrm{base}}{=}12$"]
axb.bar(x-w,early,w,color=DGRAY,zorder=3,label="early"); axb.bar(x,flat,w,color=SILVER,zorder=3,label="flat")
axb.bar(x+w,late,w,color=DEEP_RED,zorder=3,label=r"late $\gamma$2")
for i in range(2):
    axb.annotate(f"{dearly[i]:+.2f}",(x[i]-w,early[i]),textcoords="offset points",xytext=(-1.6,1.5),ha="center",fontsize=5.2,color=DGRAY)
    axb.annotate(f"{dlate[i]:+.2f}",(x[i]+w,late[i]),textcoords="offset points",xytext=(-1.6,1.5),ha="center",fontsize=5.2,color=DEEP_RED,fontweight="bold")
axb.set_xticks(x); axb.set_xticklabels(groups,fontsize=6.0)
axb.set_ylabel("15-corr top-1 (%)",fontsize=6.0,labelpad=1); axb.set_ylim(33,40.8)
axb.tick_params(axis='y',labelsize=5.4,pad=1); axb.set_title("(b) accuracy",fontsize=7.0,pad=2)
axb.grid(axis="y",alpha=0.3,zorder=0)
import os; os.makedirs("output/figures",exist_ok=True)
fig.savefig("output/figures/fig_placement.pdf",bbox_inches="tight")
fig.savefig("output/figures/fig_placement.png",dpi=200,bbox_inches="tight")
plt.close(fig)

# ---------------- appendix figure: the r_base=12 token profiles
f2,a2=plt.subplots(figsize=(3.5,2.5))
a2.plot(depth,r12_early,color=DGRAY,  lw=1.5,marker="o",ms=2.6,label=f"early {g12_early:.3f}G",zorder=3)
a2.plot(depth,r12_flat, color=SILVER, lw=1.5,marker="s",ms=2.6,label=f"flat {g12_flat:.3f}G",zorder=3)
a2.plot(depth,r12_late, color=DEEP_RED,lw=2.3,marker="o",ms=3.0,label=f"late $\\gamma$2 {g12_late:.3f}G",zorder=4)
a2.set_xlabel(r"layer depth $\ell$",fontsize=9.5); a2.set_ylabel("tokens left",fontsize=9.5)
a2.set_xticks([0,2,4,6,8,10,12]); a2.set_ylim(0,205); a2.tick_params(labelsize=8.5); a2.grid(alpha=0.3,zorder=0)
a2.legend(fontsize=8.0,loc="lower left",handlelength=1.2,handletextpad=0.4,borderpad=0.3,labelspacing=0.25,framealpha=0.85)
f2.tight_layout()
f2.savefig("output/figures/fig_placement_r12.pdf",bbox_inches="tight")
f2.savefig("output/figures/fig_placement_r12.png",dpi=200,bbox_inches="tight")
print("saved compact fig_placement.pdf (r8 curves | bars) and fig_placement_r12.pdf")
