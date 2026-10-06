"""Evaluation datasets and class masks for the benchmarks in the paper.

Dataset roots come from ``dataset_roots`` in the config. A value may contain ``${DATA_ROOT}``,
which expands to the ``DATA_ROOT`` environment variable (default ``./data``), and an environment
variable ``DATA_<KEY>`` (key upper-cased, ``-`` replaced by ``_``, e.g. ``DATA_IN_C``) overrides
the entry for that key. Keys the config leaves out fall back to ``${DATA_ROOT}/<leaf>`` with the
leaf names in ``DEFAULT_LEAVES``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
CORRUPTION_INDEX = {name: i for i, name in enumerate(CORRUPTIONS)}

# ImageNet class order is the sorted WordNet id order the classifiers were trained with.
# ImageNet-R and ImageNet-A each cover 200 of the 1000 classes; their models are evaluated on
# the logits of those classes only, following the evaluation code released with each dataset.
_MASKS = json.load(open(Path(__file__).with_name("imagenet_masks.json")))
IMAGENET_WNIDS = list(_MASKS["imagenet_wnids"])
IMAGENET_R_MASK = [w in set(_MASKS["imagenet_r_wnids"]) for w in IMAGENET_WNIDS]
IMAGENET_A_MASK = [i in set(_MASKS["imagenet_a_class_indices"]) for i in range(len(IMAGENET_WNIDS))]

# ImageNet-C-bar and ImageNet-3DCC: 224 px folders, <root>/<corruption>/<severity>/<wnid>/, a subset of the classes
CBAR_CORRUPTIONS = ["blue_noise_sample", "brownish_noise", "caustic_refraction", "checkerboard_cutout",
                    "cocentric_sine_waves", "inverse_sparkles", "perlin_noise", "plasma_noise",
                    "single_frequency_greyscale", "sparkles"]
THREEDCC_CORRUPTIONS = ["bit_error", "color_quant", "far_focus", "flash", "fog_3d", "h265_abr", "h265_crf", "iso_noise",
                        "low_light", "near_focus", "xy_motion_blur", "z_motion_blur"]

DEFAULT_LEAVES = {
    "IN": "imagenet", "IN-C": "imagenet-c", "IN-R": "imagenet-r", "IN-S": "imagenet-sketch",
    "IN-A": "imagenet-a", "IN-V2": "imagenet-v2", "nabirds-c": "nabirds-c",
    "DN126": "domainnet126",
    "IN-CBAR": "imagenet-c-bar", "IN-3DCC": "imagenet-3dcc",
}


def resolve_dataset_roots(configured: dict) -> dict:
    """Apply ``${DATA_ROOT}`` expansion, ``DATA_<KEY>`` overrides and the default leaves."""
    data_root = os.environ.get("DATA_ROOT", "./data")
    roots = {}
    for key, leaf in DEFAULT_LEAVES.items():
        value = configured.get(key, "${DATA_ROOT}/" + leaf)
        value = os.environ.get("DATA_" + key.upper().replace("-", "_"), value)
        roots[key] = value.replace("${DATA_ROOT}", data_root)
    for key, value in configured.items():
        if key not in roots:
            roots[key] = os.environ.get("DATA_" + key.upper().replace("-", "_"), value).replace("${DATA_ROOT}", data_root)
    return roots


def resolve_corruption(corruption):
    """Accept a corruption name or its index; other strings ('ori', 'all') pass through."""
    if isinstance(corruption, int):
        return corruption
    text = str(corruption)
    if text in CORRUPTION_INDEX:
        return CORRUPTION_INDEX[text]
    if text.isdigit():
        return int(text)
    return text


class ClassMaskedModel(torch.nn.Module):
    """Return only the logits of the classes a benchmark covers (as in the ImageNet-R evaluation code)."""

    def __init__(self, model, mask):
        super().__init__()
        self.model = model
        self.mask = mask

    def forward(self, x):
        return self.model(x)[:, self.mask]


def _read_pairs(path):
    with open(path) as fh:
        return [line.strip().split(" ", 1) for line in fh if line.strip()]


class NABirds(Dataset):
    """NABirds read from the metadata files of its distribution (images.txt, image_class_labels.txt,
    train_test_split.txt). Raw class ids are not contiguous; ``label_map`` renumbers them in the
    iteration order of the set of raw labels, which is the order the released checkpoints were
    trained with."""

    def __init__(self, root, train=False, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        images = _read_pairs(os.path.join(root, "images.txt"))
        labels = _read_pairs(os.path.join(root, "image_class_labels.txt"))
        split = dict(_read_pairs(os.path.join(root, "train_test_split.txt")))
        self.label_map = {k: i for i, k in enumerate(set(int(t) for _, t in labels))}
        label_of = {img_id: int(t) for img_id, t in labels}
        wanted = "1" if train else "0"
        self.samples = [(path, label_of[img_id]) for img_id, path in images if split[img_id] == wanted]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, raw = self.samples[idx]
        img = self.loader(os.path.join(self.root, "images", path))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label_map[raw]


def build_imagefolder(root):
    return ImageFolder(root=root)


def build_imagenet_subset_folder(root):
    """A class folder holding only some of the 1000 ImageNet classes: targets are the canonical indices of its wnids."""
    ds = ImageFolder(root=root)
    index = {w: i for i, w in enumerate(IMAGENET_WNIDS)}
    ds.samples = [(path, index[ds.classes[t]]) for path, t in ds.samples]
    ds.targets = [t for _, t in ds.samples]
    return ds


def build_imagenet_v2(root):
    """ImageNet-V2 folders are named by ImageNet class index; keep that index as the target."""
    ds = ImageFolder(root=root)
    ds.samples = [(path, int(ds.classes[t])) for path, t in ds.samples]
    ds.targets = [t for _, t in ds.samples]
    return ds
