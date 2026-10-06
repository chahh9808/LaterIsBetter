# Experiment drivers

`scripts/launch_table.py` covers the ImageNet-C tables; the directories here hold the drivers behind the
other blocks of numbers. They are thin layers over `main.py`: a `make_configs` script writes the configs and
a `jobs.json`, `sweep.py` runs them round-robin over GPUs, and a statistics or table script reads the
metrics back. The probes (`mechanism/`, `cleanpath/`, `agreement/`) and the `extra_corruptions/` and `tta_cost/`
scripts drive the model directly and write their own outputs.

| Directory | Paper | Run |
|---|---|---|
| `vitl_backbones/` | Table 4, ViT-L and CLIP ViT-L rows | `make_configs.py`, then `sweep.py output/vitl_backbones output/vitl_backbones/*_jobs.json` |
| `clip_backbone/` | Table 4, CLIP ViT-B row; the appendix row at the aggressive rate | `make_configs.py`, `sweep.py`, then `table4_clip.py` |
| `mae_gap/` | Table 4, MAE rows; the readout-control appendix | `run_mae.py --model {vitb,vitl} --gamma {1,2}`; `token_audit.py` |
| `natural_shift_av2/` | Table 5, ImageNet-A, V2, R and Sketch; the paired statistics appendix | `make_configs_inx.py`, `sweep.py`, `stats_av2.py`, `combine_av2.py` |
| `natural_shift_domainnet/` | Table 5, DomainNet-126; its source checkpoints | `finetune_vit_imagefolder_norm.py`, then `make_configs_dn.py`, `sweep.py`, `stats_dn.py`, `combine_dn.py` |
| `mechanism/` | Figure 2 | `probe_reduction.py`, `probe_amplification.py`, `marginal_gflops.py` |
| `cleanpath/` | The clean merge-path figure and its appendix table | `run_cleanpath.py` |
| `agreement/` | The per-image agreement figure | `run_agreement.py` |
| `diffrate/` | The searched-schedule appendix | `gen_schedules.py`, `eval_ourbudget.py`, `measure_our_overhead.py` |
| `extra_corruptions/` | The appendix on the four ImageNet-C corruptions outside the standard fifteen | `run_extra.py` |
| `tta_cost/` | The cost of one adaptation step on full tokens and on the late schedule | `flops.py` and `measure.py` per backbone and schedule, then `summarize.py` |

`vitl_backbones/vitl_calib.py` reports, for a backbone of any depth, the flat GFLOPs at each per-layer rate
and the smallest late budget at or below it; it is how the operating points of the 24-layer backbones were
set.

Environment: `DATA_ROOT` and the `DATA_<KEY>` variables for datasets (see the top-level README), `CKPT_DIR`
for checkpoints, `GPUS` and `PER_GPU` for `sweep.py`, `PY` for the interpreter, and `DIFFRATE_ROOT` for the
DiffRate clone the `diffrate/` scripts drive. Results are written under `output/`, which is not part of the
repository.
