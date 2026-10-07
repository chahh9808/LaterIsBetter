"""Shared pieces of the config generators: the repository root, the corruption list, and the
schedule blocks. The late schedule's total_r values are the iso-GFLOPs calibration of Table 1;
they hold for every 12-layer backbone at 197 tokens."""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def schedule(total_r, kind="constant", gamma=None):
    block = {"name": "tome", "total_r": total_r, "schedule": kind}
    if gamma is not None:
        block["schedule_gamma"] = gamma
    return block


# main operating point r_base=8 and the aggressive rate r_base=12, 12-layer backbones
SCHEDULES_R8 = {"none": schedule(0), "flat": schedule(8),
                "late_r8_g1": schedule(12, "late-concentrated", 1.0), "late": schedule(16, "late-concentrated", 2.0)}
SCHEDULES_R12 = {"flat_r12": schedule(12), "late_r12_g1": schedule(18, "late-concentrated", 1.0),
                 "late_r12_g2": schedule(27, "late-concentrated", 2.0)}


def load_base(name, model_name=None, **overrides):
    """A repository config with logging off, adaptation off, and the given top-level overrides."""
    cfg = json.load(open(REPO / "configs" / name))
    cfg["tta"]["enabled"] = False
    cfg["logging"]["use_wandb"] = False
    if model_name:
        cfg["model_name"] = model_name
    cfg.update(overrides)
    return cfg


def write_jobs(out_dir, tag, cells):
    """cells: dicts with dataset (key in the output name), dataset_name, split, corruption, schedule (key in
    the output name), method (block), base (config), and optionally level (default 5), seed, tta and
    checkpoint_path. Writes one config per cell under out_dir/configs and a jobs.json for sweep.py."""
    out_dir = Path(out_dir)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in cells:
        cfg = copy.deepcopy(c["base"])
        cfg["dataset_name"], cfg["dataset_split"], cfg["corruption"] = c["dataset_name"], c["split"], c["corruption"]
        cfg["level"] = c.get("level", 5)
        cfg["method"] = copy.deepcopy(c["method"])
        for key in ("seed", "checkpoint_path", "pretrained", "num_classes"):
            if key in c:
                cfg[key] = c[key]
        if "tta" in c:
            cfg["tta"] = {**cfg["tta"], **c["tta"]}
        name = f"{tag}_{c['dataset']}_{c['schedule']}"
        cfg["logging"]["run_name"] = name
        path = out_dir / "configs" / f"{name}.json"
        json.dump(cfg, open(path, "w"), indent=2)
        jobs.append({"tag": tag, "dataset": c["dataset"], "schedule": c["schedule"], "config": str(path)})
    jobs_path = out_dir / f"{tag}_jobs.json"
    json.dump(jobs, open(jobs_path, "w"), indent=2)
    print(f"wrote {len(jobs)} configs; jobs file {jobs_path}")
    return jobs_path


def cell(dataset, dataset_name, split, corruption, schedule, method, base, **extra):
    return dict(dataset=dataset, dataset_name=dataset_name, split=split, corruption=corruption,
                schedule=schedule, method=method, base=base, **extra)
