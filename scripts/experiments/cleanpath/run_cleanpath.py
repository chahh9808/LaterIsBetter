# The bipartite matching below is adapted, with modifications, from https://github.com/facebookresearch/ToMe (CC BY-NC 4.0, non-commercial use only).
"""Clean merge-path control: ToMe on a corrupted image, forced to follow the merge decisions its clean
counterpart produces.

For every image the clean version runs first and its per-block bipartite matching (unmerged, source and
destination indices) is captured; the paired ImageNet-C version then runs twice, once replaying those clean
decisions (oracle) and once with its own (own path, the ordinary run). Pairing is by loader order with a
label-equality guard. Writes output/cleanpath/<schedule>_<corruption>.json, one entry per r_base with the
clean, own-path and oracle top-1.

    python scripts/experiments/cleanpath/run_cleanpath.py --r-base 4,6,8,10 --corruptions gaussian_noise
    python scripts/experiments/cleanpath/run_cleanpath.py --schedule late-concentrated --gamma 2 --r-base 8
    python scripts/experiments/cleanpath/run_cleanpath.py --corruptions all

DeiT-S, severity 5. The late schedule takes its total_r from configs/iso_gflops_gamma_table.json.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import CORRUPTIONS, load_base  # noqa: E402
from src.config import _from_dict  # noqa: E402
from src.data import build_eval_data  # noqa: E402
from src.eval import _extract_logits  # noqa: E402
from src.methods import forward_with_method, patch_model_for_method  # noqa: E402
from src.methods import tome_extensions as TE  # noqa: E402
from src.modeling import create_model  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="constant", choices=["constant", "late-concentrated"])
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--r-base", default="4,6,8,10")
parser.add_argument("--corruptions", default="gaussian_noise", help="comma-separated names, or all")
parser.add_argument("--max-batches", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="output/cleanpath")
args = parser.parse_args()

R_BASES = [int(x) for x in args.r_base.split(",")]
CORRS = CORRUPTIONS if args.corruptions == "all" else args.corruptions.split(",")
ISO = json.load(open(REPO / "configs" / "iso_gflops_gamma_table.json"))["table"]["S"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def make_cfg(r_base, split, corruption):
    d = load_base("const_r8.json", model_name="deit_small_patch16_224", num_classes=1000, dataset_name="imagenet",
                  seed=args.seed)
    d["dataset_split"], d["corruption"], d["level"] = split, corruption, 5
    d["evaluation"] = {**d["evaluation"], "max_batches": None, "workers": 4}
    total_r = r_base if args.schedule == "constant" else ISO[str(r_base)][f"{args.gamma:.1f}"]["total_r"]
    d["method"] = {"name": "tome", "total_r": total_r, "schedule": args.schedule, "schedule_gamma": args.gamma}
    return _from_dict(d)


class Ctrl:
    mode = "own"
    store = None
    ptr = 0


ctrl = Ctrl()


def _indices(metric, r, class_token, distill_token):
    """ToMe's bipartite soft matching, returning the index triple instead of the merge closures."""
    m = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a, b = m[..., ::2, :], m[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    if class_token:
        scores[..., 0, :] = float("-inf")
    if distill_token:
        scores[..., :, 0] = float("-inf")
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx, src_idx = edge_idx[..., r:, :], edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    if class_token:
        unm_idx = unm_idx.sort(dim=1)[0]
    return unm_idx, src_idx, dst_idx


def _build_merge(unm_idx, src_idx, dst_idx, r, distill_token):
    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        onehot = F.one_hot(dst_idx[..., 0], dst.shape[-2]).to(src.dtype)
        count = onehot.sum(dim=1)
        src_sum = onehot.transpose(1, 2) @ src
        dst = (dst + src_sum) / (1.0 + count).unsqueeze(-1) if mode == "mean" else dst + src_sum
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    return merge, (lambda x: x)


def hooked_matching(metric, r, class_token=False, distill_token=False):
    """Drop-in for bipartite_soft_matching: captures, replays or simply computes the decisions."""
    protected = int(class_token) + int(distill_token)
    r = min(r, max(0, (metric.shape[1] - protected) // 2))
    if r <= 0:
        return (lambda x, mode="mean": x), (lambda x: x)
    with torch.no_grad():
        if ctrl.mode == "replay":
            unm_idx, src_idx, dst_idx, r_stored = ctrl.store[ctrl.ptr]
            ctrl.ptr += 1
            assert r_stored == r, f"replay r mismatch {r_stored} vs {r}"
        else:
            unm_idx, src_idx, dst_idx = _indices(metric, r, class_token, distill_token)
            if ctrl.mode == "capture":
                ctrl.store.append((unm_idx, src_idx, dst_idx, r))
    return _build_merge(unm_idx, src_idx, dst_idx, r, distill_token)


def unpack(batch):
    return batch[0].to(DEVICE, non_blocking=True), batch[1].to(DEVICE, non_blocking=True)


for corruption in CORRS:
    results = {}
    for r_base in R_BASES:
        set_seed(args.seed)
        cfg_corr, cfg_clean = make_cfg(r_base, "c", corruption), make_cfg(r_base, "val", corruption)
        model = create_model(cfg_corr, num_classes=1000)
        model, mcfg = patch_model_for_method(model, cfg_corr)
        model = model.to(DEVICE).eval()
        clean_loader = build_eval_data(cfg_clean).loader
        corr_loader = build_eval_data(cfg_corr).loader
        patch = TE._tome_patch()
        original = patch.bipartite_soft_matching
        patch.bipartite_soft_matching = hooked_matching
        clean = oracle = own = n = mismatch = 0
        t0 = time.time()
        try:
            with torch.no_grad():
                for bi, (cb, gb) in enumerate(zip(clean_loader, corr_loader)):
                    if args.max_batches and bi >= args.max_batches:
                        break
                    cx, cy = unpack(cb)
                    gx, gy = unpack(gb)
                    if not torch.equal(cy, gy):
                        mismatch += 1
                    ctrl.mode, ctrl.store = "capture", []
                    clean += (_extract_logits(forward_with_method(model, cx, mcfg)).argmax(1) == cy).sum().item()
                    ctrl.mode, ctrl.ptr = "replay", 0
                    oracle += (_extract_logits(forward_with_method(model, gx, mcfg)).argmax(1) == gy).sum().item()
                    ctrl.mode = "own"
                    own += (_extract_logits(forward_with_method(model, gx, mcfg)).argmax(1) == gy).sum().item()
                    n += cy.numel()
                    if bi % 80 == 0:
                        print(f"  {corruption} r{r_base} batch {bi} n={n} clean={100 * clean / n:.2f} "
                              f"own={100 * own / n:.2f} oracle={100 * oracle / n:.2f}", flush=True)
        finally:
            patch.bipartite_soft_matching = original
        res = dict(corruption=corruption, r_base=r_base, schedule=args.schedule, total_r=mcfg.total_r, n=n,
                   clean_top1=100 * clean / n, own_path_top1=100 * own / n, oracle_cleanpath_top1=100 * oracle / n,
                   oracle_minus_own=100 * (oracle - own) / n, label_mismatch_batches=mismatch,
                   sec=round(time.time() - t0))
        results[r_base] = res
        print(f"=== {corruption} r{r_base}: own={res['own_path_top1']:.2f} oracle={res['oracle_cleanpath_top1']:.2f} "
              f"({res['oracle_minus_own']:+.2f}) clean={res['clean_top1']:.2f} mismatch={mismatch} ===", flush=True)
    path = OUT / f"{args.schedule}_{corruption}.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"wrote {path}", flush=True)
