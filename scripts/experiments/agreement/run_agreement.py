"""Per-image agreement of a reduced model with the unreduced one (the data of the agreement figure).

For every image the unreduced model and the two reduced arms (flat r_base and late gamma at the iso-GFLOPs
budget) run on the same input. With c* the class the unreduced model predicts, x is the probability the
unreduced model assigns to c*, y the probability the reduced arm assigns to it, kl the divergence
KL(p_unreduced || p_reduced), and same whether the arm's top-1 is c*. Writes
output/agreement/<model>_<corruption>.npz with the arrays <condition>_<arm>_{x,y,kl,same} for the clean val
images (condition clean) and the corrupted ones (condition corr).

    python scripts/experiments/agreement/run_agreement.py                       # DeiT-S, gaussian noise, severity 5
    python scripts/experiments/agreement/run_agreement.py --corruption all      # one file per corruption
    python scripts/experiments/agreement/run_agreement.py --model vit_base_patch16_224 --corruption all

Clean images are loaded with the geometry ImageNet-C was rendered with (Resize 256, bilinear, then CenterCrop
224), so the clean and the corrupted version of an image differ only by the corruption; --clean-geometry model
uses the model's own evaluation transform instead.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from common import CORRUPTIONS, load_base, schedule  # noqa: E402
from src.config import _from_dict  # noqa: E402
from src.data import _model_norm, build_eval_data  # noqa: E402
from src.eval import _extract_logits  # noqa: E402
from src.methods import forward_with_method, patch_model_for_method  # noqa: E402
from src.modeling import create_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="deit_small_patch16_224")
parser.add_argument("--corruption", default="gaussian_noise", help="an ImageNet-C corruption, or all")
parser.add_argument("--level", type=int, default=5)
parser.add_argument("--r-base", type=int, default=8)
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--conditions", default="clean,corr")
parser.add_argument("--clean-geometry", default="inc", choices=["inc", "model"])
parser.add_argument("--max-batches", type=int, default=None)
parser.add_argument("--out", default="output/agreement")
args = parser.parse_args()

SCALE = "B" if "base" in args.model else "S"
ISO = json.load(open(REPO / "configs" / "iso_gflops_gamma_table.json"))["table"][SCALE]
TOTAL_R_LATE = ISO[str(args.r_base)][f"{args.gamma:.1f}"]["total_r"]
ARMS = {f"flat_r{args.r_base}": schedule(args.r_base),
        f"late_g{args.gamma:g}_tr{TOTAL_R_LATE}": schedule(TOTAL_R_LATE, "late-concentrated", args.gamma)}
CORRS = CORRUPTIONS if args.corruption == "all" else [args.corruption]
CONDITIONS = args.conditions.split(",")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = {"S": "const_r8.json", "B": "vitb_const_r8.json"}[SCALE]


def make_cfg(method, split, corruption):
    d = load_base(BASE, model_name=args.model, num_classes=1000, dataset_name="imagenet")
    d["dataset_split"], d["corruption"], d["level"] = split, corruption, args.level
    d["evaluation"] = {**d["evaluation"], "max_batches": None, "workers": 4}
    d["method"] = method
    return _from_dict(d)


def build(method):
    cfg = make_cfg(method, "c", CORRS[0])
    model = create_model(cfg, num_classes=1000)
    model, mcfg = patch_model_for_method(model, cfg)
    return model.to(DEVICE).eval(), mcfg


models = {"none": build(schedule(0))}
models.update({name: build(m) for name, m in ARMS.items()})


@torch.no_grad()
def probs(name, x):
    model, mcfg = models[name]
    return F.softmax(_extract_logits(forward_with_method(model, x, mcfg)).float(), dim=-1)


def collect(loader, condition):
    acc = {a: {"x": [], "y": [], "kl": [], "same": []} for a in ARMS}
    n = 0
    for bi, (x, _) in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        x = x.to(DEVICE, non_blocking=True)
        p0 = probs("none", x)
        ref = p0.argmax(-1)
        lp0 = torch.log(p0.clamp_min(1e-12))
        for a in ARMS:
            pa = probs(a, x)
            acc[a]["x"].append(p0.gather(1, ref[:, None]).squeeze(1).cpu().numpy())
            acc[a]["y"].append(pa.gather(1, ref[:, None]).squeeze(1).cpu().numpy())
            acc[a]["kl"].append((p0 * (lp0 - torch.log(pa.clamp_min(1e-12)))).sum(-1).cpu().numpy())
            acc[a]["same"].append((pa.argmax(-1) == ref).cpu().numpy())
        n += x.shape[0]
        if bi % 100 == 0:
            print(f"  {condition}: {n} images", flush=True)
    out = {f"{condition}_{a}_{k}": np.concatenate(v) for a in ARMS for k, v in acc[a].items()}
    for a in ARMS:
        print(f"{condition} {a}: n={n} KL mean {out[f'{condition}_{a}_kl'].mean():.4f} "
              f"top-1 agreement {100 * out[f'{condition}_{a}_same'].mean():.2f}%", flush=True)
    return out


clean = {}
if "clean" in CONDITIONS:
    loader = build_eval_data(make_cfg(schedule(0), "val", CORRS[0])).loader
    if args.clean_geometry == "inc":
        mean, std, _, _ = _model_norm(args.model)
        loader.dataset.transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                                              T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    clean = collect(loader, "clean")

Path(args.out).mkdir(parents=True, exist_ok=True)
for corruption in CORRS:
    out = dict(clean)
    if "corr" in CONDITIONS:
        print(f"=== {corruption} ===", flush=True)
        out.update(collect(build_eval_data(make_cfg(schedule(0), "c", corruption)).loader, "corr"))
    path = Path(args.out) / f"{args.model}_{corruption}.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path}", flush=True)
