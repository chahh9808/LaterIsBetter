"""ImageNet-C accuracy at the main operating point for three schedules on the same DiffRate model:
the schedule DiffRate searched, the flat schedule, and the late-concentrated gamma=2 schedule, at
matched THOP GFLOPs. Requires the released DiffRate implementation; point DIFFRATE_ROOT at the
clone. Data roots follow DATA_IN_C, or DATA_ROOT/imagenet-c."""
import os, sys, json, argparse, time, math
sys.path.insert(0, os.environ.get("DIFFRATE_ROOT", os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.models import create_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import DiffRate, thop

L=12; N0=197
CORRS=["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur",
       "zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression"]

def build_mergeonly(total, gamma):
    w=[((i+1)/L)**gamma for i in range(L)]; sw=sum(w)
    real=[total*x/sw for x in w]; fl=[int(x) for x in real]; rem=total-sum(fl)
    order=sorted(range(L),key=lambda i:real[i]-fl[i],reverse=True)
    for i in range(rem): fl[order[i%L]]+=1
    mk=[]; N=float(N0)
    for j in range(L):
        r=min(fl[j],int(N)-1); N=int(N-r); mk.append(N)
    return [N0]+mk[:-1], mk

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clean-json", required=True)
    ap.add_argument("--late-json", required=True)
    ap.add_argument("--data-root", default=os.environ.get("DATA_IN_C", os.path.join(os.environ.get("DATA_ROOT", "./data"), "imagenet-c")))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=create_model(a.model,pretrained=True,num_classes=1000)
    DiffRate.patch.deit(model,prune_granularity=4,merge_granularity=4); model.eval().to(device)
    sample=torch.randn(1,3,224,224).to(device)
    def thop_of(pk,mk): model.set_kept_num(pk,mk); return float(thop.profile(model,inputs=(sample,),verbose=False)[0])/1e9

    clean=json.load(open(a.clean_json)); late=json.load(open(a.late_json))
    clean_thop=thop_of(clean["prune_kept"],clean["merge_kept"])
    late_thop=thop_of(late["prune_kept"],late["merge_kept"])
    # calibrate flat (merge-only gamma0) to <= clean THOP
    lo,hi,best=0,N0-1,None
    while lo<=hi:
        mid=(lo+hi)//2; pk,mk=build_mergeonly(mid,0.0); gf=thop_of(pk,mk)
        if gf<=clean_thop+1e-6: best=(pk,mk,gf); hi=mid-1
        else: lo=mid+1
    flat_pk,flat_mk,flat_thop=best
    keys=[("clean",clean["prune_kept"],clean["merge_kept"],clean_thop),
          ("flat",flat_pk,flat_mk,flat_thop),
          ("late",late["prune_kept"],late["merge_kept"],late_thop)]
    print(f"[{a.model}] THOP: clean={clean_thop:.4f} flat={flat_thop:.4f} late={late_thop:.4f}",flush=True)
    print(f"  clean merge={clean['merge_kept']}",flush=True)
    print(f"  flat  merge={flat_mk}",flush=True)
    print(f"  late  merge={late['merge_kept']}",flush=True)

    res={k[0]:{} for k in keys}
    for corr in CORRS:
        ds=datasets.ImageFolder(os.path.join(a.data_root,corr,"5"),transform=transforms.Compose([
            transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),
            transforms.ToTensor(),transforms.Normalize(IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD)]))
        loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=8,pin_memory=True)
        correct={k[0]:0 for k in keys}; tot=0; t0=time.time()
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); tot+=y.size(0)
            for name,pk,mk,_ in keys:
                model.set_kept_num(pk,mk)
                with torch.cuda.amp.autocast(): out=model(x)
                logits=out[0] if isinstance(out,(tuple,list)) else out
                correct[name]+=(logits.argmax(1)==y).sum().item()
        for name in correct: res[name][corr]=round(100.0*correct[name]/tot,2)
        print(f"{corr:18s} "+" | ".join(f"{n}:{res[n][corr]:5.2f}" for n,_,_,_ in keys)+f"  ({time.time()-t0:.0f}s)",flush=True)
    means={k[0]:round(sum(res[k[0]].values())/len(CORRS),3) for k in keys}
    print(f"\n=== {a.model} 15-corr mean (iso-THOP ~{clean_thop:.2f}G, OUR gamma=2 operating point) ===")
    print(f"  clean={means['clean']}  flat={means['flat']}  late(g2)={means['late']}")
    print(f"  late-flat={means['late']-means['flat']:+.3f}   late-clean={means['late']-means['clean']:+.3f}")
    json.dump({"per_corruption":res,"mean":means,"thop":{k[0]:round(k[3],4) for k in keys}},open(a.out,"w"),indent=2)
    print(f"[written] {a.out}")

if __name__=="__main__":
    main()
