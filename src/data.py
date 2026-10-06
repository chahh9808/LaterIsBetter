"""Evaluation data for one run: the dataset the config names, with the model's own preprocessing."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.datasets_local import (
    CBAR_CORRUPTIONS, THREEDCC_CORRUPTIONS, build_imagenet_subset_folder,
    CORRUPTIONS, NABirds, build_imagefolder, build_imagenet_v2, resolve_corruption, resolve_dataset_roots,
)


@dataclass
class DataBundle:
    loader: DataLoader
    num_classes: int


def _model_norm(model_name: str):
    """Return (mean, std, crop_pct, interpolation) from the timm model's default_cfg."""
    try:
        import timm
        cfg = timm.create_model(model_name, pretrained=False).default_cfg
        return (cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)),
                cfg.get("crop_pct", 0.875), cfg.get("interpolation", "bilinear"))
    except Exception:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.875, "bilinear"


def _apply_model_transform(ds, model_name: str, input_size: int = 224, skip_resize: bool = False):
    """Set dataset.transform to the normalization the model expects.

    skip_resize omits Resize for sets whose images are already input_size x input_size
    (the pre-rendered corruption suites); every other set is resized through the model's crop_pct.
    """
    mean, std, crop_pct, interp = _model_norm(model_name)
    interp_code = {"bicubic": 3, "bilinear": 2, "lanczos": 1}.get(interp, 2)
    steps = []
    if not skip_resize:
        steps.append(T.Resize(int(input_size / crop_pct), interpolation=interp_code))
    steps += [T.CenterCrop(input_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    ds.transform = T.Compose(steps)


def _corruption_dir(root: str, corruption, level: int) -> str:
    idx = resolve_corruption(corruption)
    if not isinstance(idx, int) or not 0 <= idx < len(CORRUPTIONS):
        raise ValueError(f"unknown corruption {corruption!r}; expected a name or index in 0..14")
    if not 1 <= int(level) <= 5:
        raise ValueError(f"corruption level must be 1..5, got {level}")
    return os.path.join(root, CORRUPTIONS[idx], str(int(level)))


def build_eval_data(cfg) -> DataBundle:
    roots = resolve_dataset_roots(cfg.dataset_roots)
    name = cfg.dataset_name.lower()
    split = cfg.dataset_split.lower()
    corruption = resolve_corruption(cfg.corruption)
    model = cfg.model_name
    skip_resize = False

    if name == "imagenet" and split in ("val", "ori") or (name == "imagenet" and split == "c" and corruption == "ori"):
        ds, num_classes = build_imagefolder(os.path.join(roots["IN"], "val")), 1000
    elif name in ("imagenet", "imagenet-c") and split == "c":
        ds, num_classes = build_imagefolder(_corruption_dir(roots["IN-C"], corruption, cfg.level)), 1000
        skip_resize = True
    elif name == "imagenet-r" or (name == "imagenet" and split == "r"):
        ds, num_classes = build_imagefolder(roots["IN-R"]), 200
    elif name == "imagenet-sketch" or (name == "imagenet" and split == "s"):
        ds, num_classes = build_imagefolder(roots["IN-S"]), 1000
    elif name == "imagenet-a":
        ds, num_classes = build_imagefolder(roots["IN-A"]), 200
    elif name in ("imagenet-cbar", "imagenet-3dcc"):
        key, names = ("IN-CBAR", CBAR_CORRUPTIONS) if name == "imagenet-cbar" else ("IN-3DCC", THREEDCC_CORRUPTIONS)
        corr = names[corruption] if isinstance(corruption, int) else str(corruption)
        if corr not in names:
            raise ValueError(f"unknown {name} corruption {cfg.corruption!r}")
        ds = build_imagenet_subset_folder(os.path.join(roots[key], corr, str(int(cfg.level))))
        num_classes, skip_resize = 1000, True
    elif name == "imagenet-v2":
        ds, num_classes = build_imagenet_v2(roots["IN-V2"]), 1000
    elif name == "nabirds":
        ds = NABirds(_corruption_dir(roots["nabirds-c"], corruption, cfg.level), train=False)
        num_classes, skip_resize = len(ds.label_map), True
    elif name == "domainnet126":
        # The DomainNet-126 source models were finetuned with Resize(256) + CenterCrop(224) and
        # their own normalization, so evaluation uses the same pipeline.
        ds = build_imagefolder(os.path.join(roots["DN126"], "targets", f"dn_{split}", "5"))
        mean, std, _, _ = _model_norm(model)
        ds.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
        loader = DataLoader(ds, batch_size=cfg.evaluation.batch_size, shuffle=False,
                            num_workers=cfg.evaluation.workers, pin_memory=torch.cuda.is_available())
        return DataBundle(loader=loader, num_classes=len(ds.classes))
    else:
        raise ValueError(f"unsupported dataset {cfg.dataset_name!r} with split {cfg.dataset_split!r}")

    _apply_model_transform(ds, model, 224, skip_resize=skip_resize)
    loader = DataLoader(ds, batch_size=cfg.evaluation.batch_size, shuffle=False,
                        num_workers=cfg.evaluation.workers, pin_memory=torch.cuda.is_available())
    return DataBundle(loader=loader, num_classes=num_classes)
