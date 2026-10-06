from __future__ import annotations

import torch.nn as nn

from .tent import Tent
from .sar import SAR
from .eata import EATA
from .deyo import DeYO
from .foa import FOA
from .spa import SPA
from .neo import NEO


def build_tta_model(model: nn.Module, cfg) -> nn.Module:
    """Wrap the model in the test-time adaptation method the config selects."""
    if not cfg.tta.enabled:
        return model
    t = cfg.tta
    method = t.method.lower()
    if method == "tent":
        return Tent(model, steps=t.steps, lr=t.lr, momentum=t.momentum)
    if method == "sar":
        return SAR(
            model, steps=t.steps, lr=t.lr, momentum=t.momentum,
            rho=t.rho, margin_e0=t.margin_e0, reset_constant_em=t.reset_constant_em,
        )
    if method == "eata":
        return EATA(
            model, steps=t.steps, lr=t.lr, momentum=t.momentum,
            e_margin=t.e_margin, d_margin=t.d_margin, fisher_alpha=t.fisher_alpha,
        )
    if method == "deyo":
        return DeYO(
            model, steps=t.steps, lr=t.lr, momentum=t.momentum,
            deyo_margin=t.deyo_margin, margin_e0=t.margin_e0,
            plpd_threshold=t.deyo_plpd_threshold,
        )
    if method == "foa":
        return FOA(
            model,
            num_prompts=t.foa_num_prompts,
            fitness_lambda=t.foa_fitness_lambda,
            popsize=(t.foa_popsize if t.foa_popsize > 0 else None),
            source_stats_path=t.foa_source_stats_path,
            seed=cfg.seed,
        )
    if method == "spa":
        return SPA(
            model, steps=t.steps, lr=t.lr, momentum=t.momentum,
            noise_ratio=t.spa_noise_ratio, freq_mask_ratio=t.spa_freq_mask_ratio,
            predictor_lr_mult=t.spa_predictor_lr_mult,
        )
    if method == "neo":
        # NEO has no hyperparameters: the running mean is its entire state.
        return NEO(model)
    raise ValueError(f"unknown test-time adaptation method {cfg.tta.method!r}; expected tent, sar, eata, deyo, foa, spa or neo")
