"""Per-layer reduction perturbation for each reducer (the token-removal curves of the mechanism figure).

For reducer R, condition c and block l, remove RCOUNT tokens at block l only and measure the mean relative
logit deviation ||z_l - z_full|| / ||z_full|| over the first --max-batches batches. Writes
output/mechanism/<reducer>__<condition>.json with the per-block curve, its std and standard error, and n.

    python scripts/experiments/mechanism/probe_reduction.py tome
    python scripts/experiments/mechanism/probe_reduction.py evit,ats,atc,pitome   # reducers' environment

DeiT-S; conditions are clean ImageNet val and four ImageNet-C corruptions at severity 5.
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

import torch  # noqa: E402

from common import load_base  # noqa: E402
from src.config import _from_dict  # noqa: E402
from src.data import build_eval_data  # noqa: E402
from src.methods import forward_with_method, patch_model_for_method  # noqa: E402
from src.modeling import create_model  # noqa: E402

NUM_LAYERS, RCOUNT = 12, 16
RATIO = {"evit", "pitome"}  # take one cumulative keep ratio at the probed block
COUNT = {"ats", "atc"}  # take per-layer cumulative token counts over every block
CONDITIONS = [("clean", None), ("gnoise", "gaussian_noise"), ("defocus_blur", "defocus_blur"),
              ("fog", "fog"), ("jpeg_compression", "jpeg_compression")]

parser = argparse.ArgumentParser()
parser.add_argument("reducers", help="comma-separated: tome, evit, ats, atc, pitome")
parser.add_argument("--max-batches", type=int, default=16)
parser.add_argument("--layers", default="all")
parser.add_argument("--conditions", default="all")
parser.add_argument("--out", default="output/mechanism")
args = parser.parse_args()

REDUCERS = args.reducers.split(",")
LAYERS = list(range(NUM_LAYERS)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
CONDS = CONDITIONS if args.conditions == "all" else [c for c in CONDITIONS if c[0] in args.conditions.split(",")]
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_cfg(reducer, cond, loc, keep, schedule):
    d = load_base("const_r8.json", model_name="deit_small_patch16_224", num_classes=1000, dataset_name="imagenet")
    label, corr = cond
    if label == "clean":
        d["dataset_split"], d["corruption"], d["level"] = "val", "gaussian_noise", 0
    else:
        d["dataset_split"], d["corruption"], d["level"] = "c", corr, 5
    d["evaluation"] = {"batch_size": 64, "workers": 4, "max_batches": args.max_batches}
    d["method"] = {"name": reducer, "total_r": 4, "schedule": schedule,
                   "reduction_loc": list(loc), "keep_rate": list(keep)}
    return _from_dict(d)


loaders = {}
for cond in CONDS:
    loaders[cond[0]] = build_eval_data(make_cfg(REDUCERS[0], cond, [], [], "explicit")).loader
    print(f"loader[{cond[0]}] ready", flush=True)


def run_logits(model, method_cfg, loader):
    outs = []
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if bi >= args.max_batches:
                break
            o = forward_with_method(model, b[0].to(dev), method_cfg)
            o = o[0] if isinstance(o, (tuple, list)) else o
            outs.append(o.detach().float().cpu())
    return torch.cat(outs, 0)


def loc_keep(reducer, l):
    """(reduction_loc, keep_rate) that removes RCOUNT tokens at block l only; l=None keeps every token."""
    if reducer in COUNT:
        keep = [196] * NUM_LAYERS if l is None else [196] * l + [196 - RCOUNT] * (NUM_LAYERS - l)
        return list(range(NUM_LAYERS)), keep
    if l is None:
        return [], []
    return [l], [1.0 - RCOUNT / 196.0]


def build(cfg):
    model = create_model(cfg, 1000)
    model, mcfg = patch_model_for_method(model, cfg)
    return model.to(dev).eval(), mcfg


def dump(reducer, label, curve, std, n):
    sem = [s / (n ** 0.5) for s in std]
    json.dump({"reducer": reducer, "cond": label, "curve": curve, "std": std, "sem": sem, "n": n},
              open(OUT / f"{reducer}__{label}.json", "w"))
    print(f"{reducer}/{label}: curve={[round(c, 4) for c in curve]}", flush=True)


def probe_tome():
    model, mcfg = build(make_cfg("tome", CONDS[0], [], [], "constant"))
    for label, _ in CONDS:
        loader = loaders[label]
        model.r = [0] * NUM_LAYERS
        full = run_logits(model, mcfg, loader)
        fn = full.norm(dim=-1).clamp_min(1e-6)
        curve, std = [None] * NUM_LAYERS, [None] * NUM_LAYERS
        for l in LAYERS:
            r = [0] * NUM_LAYERS
            r[l] = RCOUNT
            model.r = r
            rel = (run_logits(model, mcfg, loader) - full).norm(dim=-1) / fn
            curve[l], std[l] = rel.mean().item(), rel.std().item()
        dump("tome", label, curve, std, int(full.shape[0]))


def probe_baseline(reducer):
    floc, fkeep = loc_keep(reducer, None)
    fm, fmcfg = build(make_cfg(reducer, CONDS[0], floc, fkeep, "explicit"))
    full_logits = {}
    for label, _ in CONDS:
        fl = run_logits(fm, fmcfg, loaders[label])
        full_logits[label] = (fl, fl.norm(dim=-1).clamp_min(1e-6))
    del fm
    torch.cuda.empty_cache()
    curves = {label: [None] * NUM_LAYERS for label, _ in CONDS}
    stds = {label: [None] * NUM_LAYERS for label, _ in CONDS}
    for l in LAYERS:
        loc, keep = loc_keep(reducer, l)
        m, mcfg = build(make_cfg(reducer, CONDS[0], loc, keep, "explicit"))
        for label, _ in CONDS:
            fl, fn = full_logits[label]
            rel = (run_logits(m, mcfg, loaders[label]) - fl).norm(dim=-1) / fn
            curves[label][l], stds[label][l] = rel.mean().item(), rel.std().item()
        del m
        torch.cuda.empty_cache()
    for label, _ in CONDS:
        dump(reducer, label, curves[label], stds[label], int(full_logits[label][0].shape[0]))


for reducer in REDUCERS:
    print(f"=== reducer: {reducer} ===", flush=True)
    if reducer == "tome":
        probe_tome()
    else:
        probe_baseline(reducer)
