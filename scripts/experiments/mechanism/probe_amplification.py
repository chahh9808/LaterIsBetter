"""Perturbation amplification A(l) per block (the amplification curves of the mechanism figure).

The unreduced DeiT-S (every token kept, so the forward pass is the same for every probed block) receives a
relative-norm perturbation on the patch tokens leaving block l, delta_t = alpha * ||x_t|| * u_t with u_t a
random unit vector, and the pre-logit representation z is read out. D(l) is the mean relative deviation
||z_pert - z|| / ||z|| over images and draws, and A(l) = D(l) / alpha. The same seeded directions are reused
across blocks and conditions. Writes output/mechanism/amplification.json with per-condition mean, std and
standard error over images.

    python scripts/experiments/mechanism/probe_amplification.py

Images are a seeded sample of clean ImageNet val images that also exist in ImageNet-C gaussian_noise/5, so
every condition sees the same images.
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
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from common import load_base  # noqa: E402
from src.config import _from_dict  # noqa: E402
from src.datasets_local import resolve_dataset_roots  # noqa: E402
from src.methods import patch_model_for_method  # noqa: E402
from src.modeling import create_model  # noqa: E402

MODEL_NAME = "deit_small_patch16_224"
N_BLOCKS = 12
CONDS = [("clean", None), ("gnoise", "gaussian_noise"), ("defocus_blur", "defocus_blur"),
         ("fog", "fog"), ("jpeg_compression", "jpeg_compression")]

parser = argparse.ArgumentParser()
parser.add_argument("--n-images", type=int, default=1024)
parser.add_argument("--draws", type=int, default=5)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="output/mechanism/amplification.json")
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(input_size=224):
    """(clean, corrupted): clean val images are resized and cropped as the model's timm config says;
    ImageNet-C images are already 224 px and only normalised, as in the evaluation loader."""
    cfg = timm.create_model(MODEL_NAME, pretrained=False).default_cfg
    mean, std = cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))
    interp = {"bicubic": 3, "bilinear": 2, "lanczos": 1}.get(cfg.get("interpolation", "bicubic"), 3)
    resize = int(input_size / cfg.get("crop_pct", 0.875))
    norm = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return (T.Compose([T.Resize(resize, interpolation=interp), T.CenterCrop(input_size)] + norm),
            T.Compose([T.CenterCrop(input_size)] + norm))


def build_imagelist(clean_root, inc_root, n_images, seed):
    """Seeded sample of (class/file, class index) pairs present in both clean val and gaussian_noise/5."""
    rng = random.Random(seed)
    gnoise_root = Path(inc_root) / "gaussian_noise" / "5"
    wnids = sorted(d.name for d in gnoise_root.iterdir() if d.is_dir())
    pairs = []
    for i, w in enumerate(wnids):
        cdir = Path(clean_root) / w
        if not cdir.is_dir():
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.upper() in (".JPEG", ".JPG", ".PNG") and (gnoise_root / w / f.name).exists():
                pairs.append((f"{w}/{f.name}", i))
    rng.shuffle(pairs)
    return pairs[:n_images]


class NormCapture:
    def __init__(self):
        self.out = None

    def hook(self, m, i, o):
        self.out = o.detach()


class Injector:
    def __init__(self):
        self.enabled = False
        self.delta = None

    def hook(self, m, i, out):
        if not self.enabled:
            return out
        o = out[0] if isinstance(out, tuple) else out
        o = o.clone()
        o[:, 1:, :] = o[:, 1:, :] + self.delta
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


@torch.no_grad()
def fwd_z(model, batch, norm_cap):
    model(batch)
    return model.forward_head(norm_cap.out, pre_logits=True)


def run_cond(model, injectors, norm_cap, block_out_norm, images, load):
    n = len(images)
    dev_per_image = np.zeros((N_BLOCKS, n))
    t0 = time.time()
    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, n, bs):
            chunk = images[start:start + bs]
            b = len(chunk)
            sl = slice(start, start + b)
            batch = torch.stack([load(rel) for rel, _ in chunk]).to(DEVICE)
            for inj in injectors:
                inj.enabled = False
            z_clean = fwd_z(model, batch, norm_cap)
            zcn = z_clean.norm(dim=-1).clamp_min(1e-12)
            clean_block_norm = {i: block_out_norm[i].clone() for i in range(N_BLOCKS)}
            n_patch = clean_block_norm[0].shape[1]
            feat = model.blocks[0].norm1.normalized_shape[0]
            for d in range(args.draws):
                g = torch.Generator(device=DEVICE)
                g.manual_seed(args.seed * 1_000_003 + start * 131 + d)
                u = torch.randn(b, n_patch, feat, generator=g, device=DEVICE)
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for l in range(N_BLOCKS):
                    injectors[l].delta = (args.alpha * clean_block_norm[l]).unsqueeze(-1) * u
                    for j, inj in enumerate(injectors):
                        inj.enabled = (j == l)
                    z_pert = fwd_z(model, batch, norm_cap)
                    dev_per_image[l, sl] += ((z_pert - z_clean).norm(dim=-1) / zcn).cpu().numpy()
                for inj in injectors:
                    inj.enabled = False
            if (start // bs) % 4 == 0:
                print(f"    {start + b}/{n} ({time.time() - t0:.0f}s)", flush=True)
    amp = (dev_per_image / args.draws) / args.alpha
    return amp.mean(1), amp.std(1), amp.std(1) / np.sqrt(n)


set_seed(args.seed)
cfg_dict = load_base("const_r8.json", model_name=MODEL_NAME, num_classes=1000)
cfg_dict["method"] = {"name": "tome", "total_r": 0, "schedule": "constant", "schedule_gamma": 1.0}
cfg = _from_dict(cfg_dict)
model = create_model(cfg, 1000)
model, _ = patch_model_for_method(model, cfg)
model = model.to(DEVICE).eval()
model.r = [0] * N_BLOCKS
model._tome_base_r = [0] * N_BLOCKS

roots = resolve_dataset_roots(cfg_dict["dataset_roots"])
clean_root, inc_root = str(Path(roots["IN"]) / "val"), roots["IN-C"]
clean_tf, corrupt_tf = build_transforms()
images = build_imagelist(clean_root, inc_root, args.n_images, args.seed)
print("N images:", len(images), flush=True)

norm_cap = NormCapture()
model.norm.register_forward_hook(norm_cap.hook)
injectors = [Injector() for _ in range(N_BLOCKS)]
for i, blk in enumerate(model.blocks):
    blk.register_forward_hook(injectors[i].hook)
block_out_norm = {}


def record(idx):
    def rec(m, i, o):
        oo = o[0] if isinstance(o, tuple) else o
        block_out_norm[idx] = oo[:, 1:, :].norm(dim=-1).detach()
    return rec


for i, blk in enumerate(model.blocks):
    blk.register_forward_hook(record(i))

out = {}
for label, corr in CONDS:
    print(f"=== A(l) condition={label} ===", flush=True)
    if corr is None:
        def load(rel):
            return clean_tf(Image.open(Path(clean_root) / rel).convert("RGB"))
    else:
        def load(rel, corr=corr):
            return corrupt_tf(Image.open(Path(inc_root) / corr / "5" / rel).convert("RGB"))
    mean, std, sem = run_cond(model, injectors, norm_cap, block_out_norm, images, load)
    out[label] = {"mean": [round(x, 4) for x in mean.tolist()], "std": [round(x, 4) for x in std.tolist()],
                  "sem": [round(x, 5) for x in sem.tolist()], "N": len(images)}
    print(f"  mean: {out[label]['mean']}", flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(f"wrote {args.out}")
