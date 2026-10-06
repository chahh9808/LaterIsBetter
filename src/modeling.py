from __future__ import annotations

import timm
import torch

from src.baselines import BASELINE_METHODS, create_baseline_model


def create_model(cfg, num_classes: int):
    method_name = cfg.method.name.lower()

    if method_name in BASELINE_METHODS:
        return create_baseline_model(cfg, num_classes)

    model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=num_classes)
    if cfg.checkpoint_path:
        state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict"):
            if isinstance(state, dict) and key in state and "head.weight" not in state:
                state = state[key]
        model.load_state_dict(state, strict=True)
    model.eval()
    return model
