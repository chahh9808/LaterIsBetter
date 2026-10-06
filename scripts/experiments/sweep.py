"""Run the jobs a make_configs script wrote, round-robin over GPUs.

    python scripts/experiments/sweep.py <out_dir> <jobs.json> [<jobs.json> ...] [--max-batches N]

GPUS (default "0") lists the CUDA devices, PER_GPU (default 1) the concurrent jobs per device, and
PY the interpreter (default: the one running this script). Each job runs main.py on its config and
writes <out_dir>/<tag>_<dataset>_<schedule>.json, the per-image correctness to <out_dir>/preds/,
and a log to <out_dir>/logs/. Jobs whose metrics file already exists are skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("jobs", nargs="+", help="jobs.json files written by a make_configs script")
    ap.add_argument("--max-batches", type=int, default=None, help="limit each run (smoke tests)")
    args = ap.parse_args()
    py = os.environ.get("PY", sys.executable)
    gpus = os.environ.get("GPUS", "0").split(",")
    per_gpu = int(os.environ.get("PER_GPU", "1"))
    out = Path(args.out_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    jobs = [j for f in args.jobs for j in json.load(open(f))]
    pending = [j for j in jobs if not (out / f"{j['tag']}_{j['dataset']}_{j['schedule']}.json").exists()]
    print(f"{len(pending)} pending / {len(jobs)} total", flush=True)
    slots = {g: [] for g in gpus}
    t0 = time.time()
    failed = []
    while pending or any(slots.values()):
        for g in gpus:
            slots[g] = [p for p in slots[g] if p.poll() is None]
            while pending and len(slots[g]) < per_gpu:
                j = pending.pop(0)
                name = f"{j['tag']}_{j['dataset']}_{j['schedule']}"
                cmd = [py, str(REPO / "main.py"), "--config", j["config"], "--device", "cuda",
                       "--metrics-json", str(out / f"{name}.json"), "--dump-preds", str(out / "preds" / f"{name}.npy")]
                if args.max_batches:
                    cmd += ["--max-batches", str(args.max_batches)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / "logs" / f"{name}.log", "w"), stderr=subprocess.STDOUT)
                p.job_name = name
                slots[g].append(p)
                print(f"[{time.time() - t0:6.0f}s] start {name} on gpu{g}", flush=True)
        for g in gpus:
            for p in list(slots[g]):
                if p.poll() is not None:
                    status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                    if p.returncode != 0:
                        failed.append(p.job_name)
                    print(f"[{time.time() - t0:6.0f}s] done  {p.job_name}: {status}", flush=True)
                    slots[g].remove(p)
        time.sleep(2)
    print(f"finished in {time.time() - t0:.0f}s; failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
