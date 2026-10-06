from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MethodConfig:
    name: str = "tome"
    total_r: int = 16
    schedule: str = "constant"
    schedule_beta: float = 1.0
    keep_rate: List[float] = field(default_factory=lambda: [0.7])
    reduction_loc: List[int] = field(default_factory=list)
    cluster_iters: int = 5
    k_neighbors: int = 5
    not_contiguous: bool = False
    min_radius: int = 2
    schedule_breakpoints: List[int] = field(default_factory=list)
    schedule_weights: List[float] = field(default_factory=list)
    schedule_gamma: float = 2.0
    linkage: str = "average"
    cluster_backend: str = "sklearn"
    proportional_attn: bool = True
    pitome_use_bsm_pitome: bool = False


_LN1000 = math.log(1000)


@dataclass
class TTAConfig:
    enabled: bool = False
    method: str = "tent"
    steps: int = 1
    lr: float = 1e-3                        # default matches Tent official lr; per-method ctors may override
    momentum: float = 0.9                   # SGD momentum (matches all official TTA implementations)
    # SAR / SAM
    rho: float = 0.05
    margin_e0: float = 0.4 * _LN1000       # ~2.763  (reliable-sample entropy threshold)
    reset_constant_em: float = 0.2          # EMA reset threshold
    # EATA
    e_margin: float = 0.4 * _LN1000        # ~2.763, the official default
    d_margin: float = 0.05                  # cosine-similarity redundancy threshold
    fisher_alpha: float = 2000.0
    # DeYO
    deyo_margin: float = 0.5 * _LN1000     # ~3.454  (first-pass entropy filter)
    deyo_plpd_threshold: float = 0.1       # PLPD drop threshold
    # FOA (Niu et al., ICML 2024): CMA-ES over prompt tokens
    foa_num_prompts: int = 3               # paper default
    foa_fitness_lambda: float = 0.4        # paper default (0.2 for ImageNet-R)
    foa_popsize: int = 0                   # 0 → auto: int(4 + 3·log(dim))
    foa_source_stats_path: str = ""        # cached layer-wise CLS stats (.pt); empty → entropy-only fallback
    # SPA (Niu et al., ICML 2025): consistency over a masked and a noisy view
    spa_noise_ratio: float = 0.4           # average high-frequency noise fraction, the ViT default
    spa_freq_mask_ratio: float = 0.2       # low-frequency amplitudes dropped, the ViT default
    spa_predictor_lr_mult: float = 5.0     # predictor lr over norm lr, the official 0.05 / 0.01


@dataclass
class EvalConfig:
    batch_size: int = 64
    workers: int = 4
    max_batches: Optional[int] = None
    dump_predictions: str = ""  # .npy path; writes per-image correctness for paired statistics


@dataclass
class EfficiencyConfig:
    cpu_warmup: int = 10
    cpu_iters: int = 50
    gpu_warmup: int = 20
    gpu_iters: int = 100
    compute_cpu_latency: bool = False
    compute_gpu_latency: bool = False
    compute_gflops: bool = True


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project: str = "token-reduction-schedules"
    run_name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)




@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    model_name: str = "deit_small_patch16_224"
    pretrained: bool = True
    checkpoint_path: str = ""  # finetuned weights to load into the model, when the benchmark needs them
    num_classes: int = 1000

    dataset_name: str = "imagenet"
    dataset_split: str = "c"
    corruption: str = "0"
    level: int = 5

    dataset_roots: Dict[str, str] = field(default_factory=dict)

    method: MethodConfig = field(default_factory=MethodConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def _build(section_cls, values: Dict[str, Any]):
    """Build a config section from the keys it declares, reporting the rest.

    A config may carry settings this section does not have; dropping them keeps it runnable
    instead of failing on an unexpected argument.
    """
    known = {f.name for f in fields(section_cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        print(f"[config] ignoring {len(unknown)} setting(s) {section_cls.__name__} does not have: "
              f"{', '.join(unknown[:6])}{' ...' if len(unknown) > 6 else ''}")
    return section_cls(**{k: v for k, v in values.items() if k in known})


def _from_dict(cfg: Dict[str, Any]) -> ExperimentConfig:
    method = _build(MethodConfig, cfg.get("method", {}))
    tta = _build(TTAConfig, cfg.get("tta", {}))
    evaluation = _build(EvalConfig, cfg.get("evaluation", {}))
    efficiency = _build(EfficiencyConfig, cfg.get("efficiency", {}))
    logging = _build(LoggingConfig, cfg.get("logging", {}))

    kwargs = {
        k: v
        for k, v in cfg.items()
        if k not in {"method", "tta", "evaluation", "efficiency", "logging", "comparison"}
    }
    return ExperimentConfig(
        **kwargs,
        method=method,
        tta=tta,
        evaluation=evaluation,
        efficiency=efficiency,
        logging=logging,
    )


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if overrides:
        raw = _merge(raw, overrides)

    return _from_dict(raw)
