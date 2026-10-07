import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
GEN="output/mechanism"
AJS="output/mechanism/amplification.json"
STH="output/mechanism/marginal_gflops.json"
DR="#750014";BK="#000000";BL="#1966ff";GR="#00ad00";PU="#9933ff";OR="#e07b00";SBLUE="#1a6faf"
plt.rcParams.update({"font.size":15,"axes.titlesize":15.5,"axes.labelsize":15,"xtick.labelsize":12.5,"ytick.labelsize":13,"legend.fontsize":11.5})
EVIT="#8c564b"  # brown, distinct from blue S-curve / late shade
REDS=[("tome","ToMe",DR,"^"),("evit","EViT",EVIT,"s"),("ats","ATS",GR,"D"),("atc","ATC",PU,"v"),("pitome","PiToMe",OR,"o")]
Cm={r:json.load(open(f"{GEN}/{r}__gnoise.json"))["curve"] for r,*_ in REDS}
Cmn={r:[Cm[r][i]/Cm[r][0] for i in range(12)] for r,*_ in REDS}
CONDS=[("clean","clean",BK,"--","o",2.6,6),("gnoise","gauss. noise",DR,"-","^",2.9,5),
       ("defocus_blur","defocus blur","#b11226","-","s",2.0,4),("fog","fog","#e03131","-","D",2.0,4),
       ("jpeg_compression","jpeg comp.","#f0908c","-","v",2.0,4)]
A={t:json.load(open(AJS))[t]["mean"] for t,*_ in CONDS}
Sth=json.load(open(STH))["S_thop"]; Sn=[s/Sth[0] for s in Sth]   # THOP-exact marginal GFLOPs saved, normalized
Ctome=Cmn["tome"]
x=list(range(1,13)); fig,axs=plt.subplots(1,3,figsize=(15.0,4.3))
# (a) C_m curves, 5 reducers
for j,(k,lab,c,mk) in enumerate(REDS): axs[0].plot(x,Cmn[k],"-",color=c,marker=mk,ms=5.6,lw=2.3,label=lab,zorder=10-j)
axs[0].axvspan(0.5,4.5,color="#cf1020",alpha=0.06);axs[0].axvspan(8.5,12.5,color="#1966ff",alpha=0.05)
axs[0].set_xlabel(r"layer index $\ell$");axs[0].set_ylabel(r"$D(\ell)/D(1)$");axs[0].set_title("(a) reduction distortion (gauss. noise)")
axs[0].set_xticks([1,4,8,12]);axs[0].set_xlim(0.5,12.5);axs[0].set_ylim(bottom=0);axs[0].grid(alpha=0.3);axs[0].legend(loc="upper right",bbox_to_anchor=(1.0,0.82),fontsize=11)
ya=axs[0].get_ylim()[1];axs[0].text(2.5,ya*0.92,"early",ha="center",color="#750014",fontsize=12,fontweight="bold");axs[0].text(10.5,ya*0.92,"late",ha="center",color="#1966ff",fontsize=12,fontweight="bold")
# (b) A
for k,lab,c,ls,mk,lw,z in CONDS: axs[1].plot(x,A[k],ls=ls,color=c,marker=mk,ms=5.6,lw=lw,label=lab,zorder=z)
axs[1].axvspan(0.5,4.5,color="#cf1020",alpha=0.06);axs[1].axvspan(8.5,12.5,color="#1966ff",alpha=0.05)
axs[1].set_xlabel(r"layer index $\ell$");axs[1].set_ylabel(r"amplification $A(\ell)$");axs[1].set_title(r"(b) amplification $A(\ell)$")
axs[1].set_xticks([1,4,8,12]);axs[1].set_xlim(0.5,12.5);axs[1].set_ylim(bottom=0);axs[1].grid(alpha=0.3)
axs[1].legend(loc="upper right",fontsize=11)
# (c) error vs compute trade-off (THOP-exact S)
axs[2].plot(x,Ctome,"-",color=DR,marker="^",ms=5.6,lw=2.7,label=r"ToMe distortion $D(\ell)$",zorder=6)
axs[2].plot(x,Sn,"-",color=SBLUE,marker="o",ms=5.6,lw=2.7,label=r"compute saved $S(\ell)$",zorder=5)
axs[2].axvspan(0.5,4.5,color="#cf1020",alpha=0.06);axs[2].axvspan(8.5,12.5,color="#1966ff",alpha=0.05)
axs[2].set_xlabel(r"layer index $\ell$");axs[2].set_ylabel(r"normalized to $\ell{=}1$");axs[2].set_title("(c) distortion vs. compute trade-off")
axs[2].set_xticks([1,4,8,12]);axs[2].set_xlim(0.5,12.5);axs[2].set_ylim(0,1.06);axs[2].grid(alpha=0.3);axs[2].legend(loc="upper right",fontsize=11)
fig.tight_layout()
out="output/figures/fig_mechtoken.pdf"
fig.savefig(out,bbox_inches="tight");fig.savefig("output/figures/fig_mechtoken.png",dpi=150,bbox_inches="tight");plt.close(fig)
print("saved 3-panel (a)D curves (b)A (c)THOP trade-off ->",out)
