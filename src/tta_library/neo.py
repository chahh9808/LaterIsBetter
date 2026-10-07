# Reimplementation of NEO (Murphy et al., ICLR 2026); original repository, which has no license file:
# https://github.com/awesomealex1/NEO
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.param_utils import _find_classifier


class NEO(nn.Module):
    """NEO: latent re-centring, no optimization (Murphy et al., ICLR 2026).
    Official repo: https://github.com/awesomealex1/NEO

    The whole method is a running mean of the classifier's input over the test stream. Each batch
    updates that mean and is then classified from its own feature minus the mean, which moves the
    shifted embeddings back onto the origin the classifier was trained around. Nothing is optimised:
    no gradients, no optimizer, no learning rate, and a step costs one forward pass, so the method
    is as cheap as inference.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in model.parameters():
            p.requires_grad_(False)

        classifier = _find_classifier(model)
        # Registered so .to(device) carries it with the model, as the official code has to do by hand.
        self.register_buffer("center", torch.zeros(classifier.in_features))
        self.seen = 0
        classifier.register_forward_pre_hook(self._recentre)

    def _recentre(self, module: nn.Module, inputs: tuple):
        """Fold this batch into the running mean, then hand the classifier the centred feature.

        Hooking the classifier's input is what the official NEO layer does by subclassing
        ``nn.Linear``; as a hook it leaves the model's own forward in place, which is what resets
        the token-merging schedule per call.
        """
        z = inputs[0]
        n = z.shape[0]
        self.seen += n
        # The cumulative mean over everything the stream has shown, the official update.
        center = self.center.to(z)
        self.center = (self.seen - n) / self.seen * center + n / self.seen * z.mean(0)
        return (z - self.center,)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    def reset(self) -> None:
        self.seen = 0
        self.center = torch.zeros_like(self.center)
