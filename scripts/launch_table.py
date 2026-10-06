"""Write the configs and job list for one of the paper's tables, to be run with scripts/experiments/sweep.py.

    python scripts/launch_table.py table1                       # Table 1: DeiT-S, six rates, six exponents
    python scripts/launch_table.py table2                       # Table 2: severities 1-5 and clean, flat vs late gamma=1/2
    python scripts/launch_table.py table3 --reducer tome        # Table 3 (gamma=2) and its gamma=1 appendix: one reducer, DeiT-S/B
    python scripts/launch_table.py table4                       # Table 4 and its r_base=12 appendix: DeiT-B, ViT-S, ViT-B
    python scripts/launch_table.py table6                       # Table 6 and the ViT-B r_base=4 appendix: five adaptation methods, 3 seeds
    python scripts/launch_table.py table6_spa                   # the SPA row of Table 6 on its own job list, into output/table6
    python scripts/launch_table.py table6_neo                   # likewise the NEO row
    python scripts/launch_table.py table6_full                  # Table 6's full-token reference: the same methods, no reduction
    python scripts/launch_table.py nabirds --backbone ViT-S --checkpoint <pth>   # NaBirds-C (finetuned DeiT-S or ViT-S)
    python scripts/launch_table.py teaser                       # Figure 1: DeiT-S, rates 0-12, clean and gaussian noise
    python scripts/launch_table.py uncompressed                 # Figure 5 and its appendix table: DeiT-S, no reduction, 15 corruptions
    python scripts/launch_table.py placement                    # Figure 3: the early-concentrated mirror schedule, r_base 8/12
    python scripts/launch_table.py shapes                       # appendix: exponential and step late shapes at r_base 8
    python scripts/launch_table.py shiftsuites                  # Table 5 rows and appendix: ImageNet-C-bar and 3DCC, DeiT-S/ViT-S
    python scripts/experiments/sweep.py output/<table> output/<table>/*_jobs.json
    python scripts/summarize_table.py <table> output/<table>

The late budgets come from configs/iso_gflops_gamma_table.json, the iso-GFLOPs calibration. The
four reducers other than ToMe need the second environment (requirements-reducers.txt); run their
job lists with PY set to that interpreter.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from common import CORRUPTIONS, cell, load_base, schedule, write_jobs  # noqa: E402
from src.datasets_local import CBAR_CORRUPTIONS, THREEDCC_CORRUPTIONS  # noqa: E402

ISO = json.load(open(REPO / "configs" / "iso_gflops_gamma_table.json"))["table"]
MODEL = {"DeiT-S": ("deit_small_patch16_224", "S"), "DeiT-B": ("deit_base_patch16_224", "B"),
         "ViT-S": ("vit_small_patch16_224", "S"), "ViT-B": ("vit_base_patch16_224", "B")}
BASE = {"DeiT-S": "const_r8.json", "DeiT-B": "deitb_const_r8.json", "ViT-S": "vits_const_r8.json", "ViT-B": "vitb_const_r8.json"}
GAMMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
REDUCER_BASE = {"tome": None, "evit": "baselines/evit.json", "ats": "baselines/ats.json", "atc": "baselines/atc.json", "pitome": "baselines/pitome.json"}
# Table 6 learning rates, fixed per method and backbone across the two schedules. SPA's official
# rate is 1e-2; it diverges on both backbones here, so both run at 5e-4, the largest rate whose
# flat baseline survives every corruption and seed (the stability protocol, measured at full
# length: at 1e-2 the ViT-B flat baseline collapses in 8 of 15 corruption x seed cells, at 1e-3 in
# 1, at 5e-4 in none).
LR = {("ViT-S", "tent"): 5e-4, ("ViT-S", "sar"): 5e-4, ("ViT-S", "eata"): 1e-4, ("ViT-S", "spa"): 5e-4,
      ("ViT-B", "tent"): 1e-4, ("ViT-B", "sar"): 5e-4, ("ViT-B", "eata"): 1e-3, ("ViT-B", "spa"): 5e-4}
SEEDS = [42, 43, 44]
# Table 3 rates and exponents per backbone; gamma=1 at r_base 6/8/10 is the appendix table
RATES3 = {"DeiT-S": {4: (2.0,), 6: (1.0, 2.0), 8: (1.0, 2.0), 10: (1.0,)}, "DeiT-B": {4: (2.0,), 6: (2.0,), 8: (2.0,)}}
# Table 6 rates per backbone; r_base=4 is the appendix table, which reports both backbones there
RATES6 = {"ViT-S": (4, 8), "ViT-B": (4, 8)}
METHODS6 = ("tent", "eata", "sar", "foa", "spa", "neo")
# Tent, SAR and NEO have no seed-dependent randomness, so their full-token reference runs one seed
STOCHASTIC6 = ("eata", "foa", "spa")
# early-concentrated gamma=2 budgets whose GFLOPs are at or just above flat, so early never has less compute
EARLY_TOTAL_R = {8: 5, 12: 8}
# alternative late shapes, each calibrated to GFLOPs at or below flat r_base=8
SHAPES = {"exp_b3": {"name": "tome", "total_r": 15, "schedule": "exponential", "schedule_beta": 3.0},
          "step_w12": {"name": "tome", "total_r": 14, "schedule": "piecewise-increasing",
                       "schedule_breakpoints": [6, 6], "schedule_weights": [1, 1, 12]}}
SUITES = {"cbar": ("imagenet-cbar", CBAR_CORRUPTIONS), "3dcc": ("imagenet-3dcc", THREEDCC_CORRUPTIONS)}
# targets whose runs belong in another target's directory, so one summary covers them together
OUT_DIR = {"table6_spa": "table6", "table6_neo": "table6"}


def late(scale, r_base, gamma):
    return schedule(ISO[scale][str(r_base)][f"{gamma:.1f}"]["total_r"], "late-concentrated", gamma)


def corruptions(base, sched, level=5, prefix="inc"):
    for i, c in enumerate(CORRUPTIONS):
        for key, method in sched.items():
            yield cell(f"{prefix}_{c}", "imagenet", "c", str(i), key, method, base, level=level)


def table1(out):
    name, scale = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    sched = {}
    for r in (2, 4, 6, 8, 10, 12):
        sched[f"r{r}_flat"] = schedule(r)
        for g in GAMMAS:
            sched[f"r{r}_g{g:g}"] = late(scale, r, g)
    write_jobs(out, "table1", list(corruptions(base, sched)))


def table2(out):
    name, scale = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    sched = {"flat": schedule(8), "g1": late(scale, 8, 1.0), "g2": late(scale, 8, 2.0)}
    cells = [cell("clean", "imagenet", "val", "ori", k, m, base) for k, m in sched.items()]
    for level in (1, 2, 3, 4, 5):
        cells += list(corruptions(base, {f"sev{level}_{k}": m for k, m in sched.items()}, level=level))
    write_jobs(out, "table2", cells)


def table3(out, reducer):
    cells = []
    for backbone, rates in RATES3.items():
        name, scale = MODEL[backbone]
        base = load_base(REDUCER_BASE[reducer] or BASE[backbone], model_name=name)
        for r, gammas in rates.items():
            for key, method in [(f"r{r}_flat", schedule(r))] + [(f"r{r}_g{g:g}", late(scale, r, g)) for g in gammas]:
                if reducer != "tome":
                    method = {**base["method"], "total_r": method["total_r"], "schedule": method["schedule"],
                              **({"schedule_gamma": method["schedule_gamma"]} if "schedule_gamma" in method else {})}
                cells += [dict(c, dataset=f"{backbone}_{c['dataset']}") for c in corruptions(base, {key: method})]
    write_jobs(out, f"table3_{reducer}", cells)


def table4(out):
    """Table 4 and its r_base=12 appendix: DeiT-B, ViT-S and ViT-B at flat, late gamma=1 and gamma=2 (DeiT-S is Table 1)."""
    cells = []
    for backbone in ("DeiT-B", "ViT-S", "ViT-B"):
        name, scale = MODEL[backbone]
        base = load_base(BASE[backbone], model_name=name)
        for r in (8, 12):
            sched = {f"r{r}_flat": schedule(r), f"r{r}_g1": late(scale, r, 1.0), f"r{r}_g2": late(scale, r, 2.0)}
            cells += [dict(c, dataset=f"{backbone}_{c['dataset']}") for c in corruptions(base, sched)]
    write_jobs(out, "table4", cells)


def table6(out, methods=METHODS6, tag="table6"):
    cells = []
    for backbone, rates in RATES6.items():
        name, scale = MODEL[backbone]
        base = load_base(BASE[backbone], model_name=name)
        for method in methods:
            tta = {"enabled": True, "method": method, "steps": 1,
                   "foa_source_stats_path": str(out / f"foa_source_stats_{name}.pt")}
            if LR.get((backbone, method)) is not None:
                tta["lr"] = LR[(backbone, method)]
            for seed in SEEDS:
                for r in rates:
                    for key, m in (("flat", schedule(r)), ("late", late(scale, r, 2.0))):
                        cells += [dict(c, dataset=f"{backbone}_{c['dataset']}", seed=seed, tta=tta)
                                  for c in corruptions(base, {f"{method}_r{r}_{key}_s{seed}": m})]
    write_jobs(out, tag, cells)


def table6_full(out):
    """Table 6's full-token reference: each method at the same rate and seeds, on 197 tokens in every block.
    FOA's source statistics depend on the token profile, so they are kept apart from Table 6's."""
    cells = []
    for backbone in RATES6:
        name, _ = MODEL[backbone]
        base = load_base(BASE[backbone], model_name=name)
        for method in METHODS6:
            tta = {"enabled": True, "method": method, "steps": 1,
                   "foa_source_stats_path": str(out / f"foa_source_stats_{name}_full.pt")}
            if LR.get((backbone, method)) is not None:
                tta["lr"] = LR[(backbone, method)]
            for seed in SEEDS if method in STOCHASTIC6 else SEEDS[:1]:
                cells += [dict(c, dataset=f"{backbone}_{c['dataset']}", seed=seed, tta=tta)
                          for c in corruptions(base, {f"{method}_full_s{seed}": schedule(0)})]
    write_jobs(out, "table6_full", cells)


def nabirds(out, checkpoint, backbone):
    name, scale = MODEL[backbone]
    base = load_base(BASE[backbone], model_name=name, pretrained=False, num_classes=555, checkpoint_path=checkpoint)
    sched = {"flat": schedule(8), "g2": late(scale, 8, 2.0)}
    cells = [cell(f"{backbone}_nabirdsc_{c}", "nabirds", "c", str(i), k, m, base)
             for i, c in enumerate(CORRUPTIONS) for k, m in sched.items()]
    write_jobs(out, f"nabirds_{backbone}", cells)


def teaser(out):
    """Figure 1: accuracy against the rate for the flat and the late schedule, clean and gaussian noise."""
    name, scale = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    sched = {"r0": schedule(0)}
    for r in (2, 4, 6, 8, 10, 12):
        sched[f"r{r}"] = schedule(r)
        sched[f"late_r{r}"] = late(scale, r, 2.0)
    cells = [cell("clean", "imagenet", "val", "ori", k, m, base) for k, m in sched.items()]
    cells += [cell("gaussian_noise", "imagenet", "c", "0", k, m, base) for k, m in sched.items()]
    write_jobs(out, "teaser", cells)


def uncompressed(out):
    """Figure 5 and its appendix table: the DeiT-S model with no reduction on every corruption."""
    name, _ = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    write_jobs(out, "uncompressed", list(corruptions(base, {"r0": schedule(0)})))


def placement(out):
    """Figure 3: the early-concentrated mirror of the late schedule at r_base 8 and 12 (flat and late are Table 1 runs)."""
    name, scale = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    sched = {f"early_r{r}": schedule(tr, "early-concentrated", 2.0) for r, tr in EARLY_TOTAL_R.items()}
    write_jobs(out, "placement", list(corruptions(base, sched)))


def shapes(out):
    """Appendix: exponential and two-level step late shapes at r_base=8 (flat and the power law are Table 1 runs)."""
    name, scale = MODEL["DeiT-S"]
    base = load_base(BASE["DeiT-S"], model_name=name)
    write_jobs(out, "shapes", list(corruptions(base, SHAPES)))


def shiftsuites(out):
    """Table 5 rows and appendix: ImageNet-C-bar and ImageNet-3DCC at severity 5, DeiT-S and ViT-S, flat vs late gamma=2."""
    cells = []
    for backbone in ("DeiT-S", "ViT-S"):
        name, scale = MODEL[backbone]
        base = load_base(BASE[backbone], model_name=name)
        sched = {"flat": schedule(8), "g2": late(scale, 8, 2.0)}
        for suite, (dataset_name, corrs) in SUITES.items():
            cells += [cell(f"{backbone}_{suite}_{c}", dataset_name, "c", c, key, m, base)
                      for c in corrs for key, m in sched.items()]
    write_jobs(out, "shiftsuites", cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=["table1", "table2", "table3", "table4", "table6", "table6_spa", "table6_neo", "table6_full", "nabirds", "teaser", "uncompressed", "placement", "shapes", "shiftsuites"])
    ap.add_argument("--reducer", choices=sorted(REDUCER_BASE), default="tome")
    ap.add_argument("--checkpoint", help="finetuned weights for the NaBirds-C row")
    ap.add_argument("--backbone", choices=["DeiT-S", "ViT-S"], default="ViT-S", help="backbone of the NaBirds-C row")
    ap.add_argument("--out", default=None, help="output directory (default output/<table>)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else REPO / "output" / OUT_DIR.get(a.table, a.table)
    if a.table == "table1": table1(out)
    elif a.table == "table2": table2(out)
    elif a.table == "table3": table3(out, a.reducer)
    elif a.table == "table4": table4(out)
    elif a.table == "table6": table6(out)
    elif a.table == "table6_spa": table6(out, methods=("spa",), tag="table6_spa")
    elif a.table == "table6_neo": table6(out, methods=("neo",), tag="table6_neo")
    elif a.table == "table6_full": table6_full(out)
    elif a.table == "teaser": teaser(out)
    elif a.table == "uncompressed": uncompressed(out)
    elif a.table == "placement": placement(out)
    elif a.table == "shapes": shapes(out)
    elif a.table == "shiftsuites": shiftsuites(out)
    else:
        if not a.checkpoint: sys.exit("nabirds needs --checkpoint")
        nabirds(out, a.checkpoint, a.backbone)
    print(f"next: python scripts/experiments/sweep.py {out} {out}/*_jobs.json")


if __name__ == "__main__":
    main()
