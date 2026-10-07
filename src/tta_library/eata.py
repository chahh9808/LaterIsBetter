# Adapted, with modifications, from https://github.com/mr-eggplant/EATA (MIT License, Copyright (c) 2023 Shuaicheng Niu et al.).
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils.entropy import _softmax_entropy
from src.utils.param_utils import _collect_ln_params, _configure_norm_only


class EATA(nn.Module):
    """EATA: efficient entropy-min with sample selection + cosine redundancy filter (Niu et al., ICML 2022).
    Official repo: https://github.com/mr-eggplant/EATA
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
        lr: float = 1e-3,
        momentum: float = 0.9,
        e_margin: float = 0.4 * math.log(1000),
        d_margin: float = 0.05,
        fisher_alpha: float = 2000.0,
    ):
        super().__init__()
        self.model = model
        self.steps = max(1, int(steps))
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self._current_probs: torch.Tensor | None = None

        params = _configure_norm_only(model, _collect_ln_params)
        self.optimizer = (
            torch.optim.SGD(params, lr=lr, momentum=momentum) if params else None
        )

        # Online EWC: initialized lazily on first forward (model is on GPU by then).
        self._params0: dict | None = None
        self._fishers: dict | None = None

    @staticmethod
    def _ema_probs(current, new_probs: torch.Tensor):
        if not new_probs.numel():
            return current
        mean = new_probs.mean(0)
        return mean if current is None else 0.9 * current + 0.1 * mean

    @torch.enable_grad()
    def compute_fisher(self, loader, fisher_size: int = 2000) -> None:
        """Estimate Fisher diagonal from exactly `fisher_size` clean source samples.

        Matches official EATA (Niu et al., ICML 2022): pseudo-label cross-entropy
        over `fisher_size` clean samples (default 2000), accumulate grad^2 per
        batch, divide by the number of batches. The last batch is truncated so
        exactly `fisher_size` samples are used. Call once after model.to(device),
        before the TTA loop.
        """
        device = next(self.model.parameters()).device
        self._params0 = {n: p.detach().clone()
                         for n, p in self.model.named_parameters() if p.requires_grad}
        fisher_acc = {n: torch.zeros_like(p)
                      for n, p in self.model.named_parameters() if p.requires_grad}
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        n_samples = 0
        n_batches = 0
        self.model.train()
        for x, _ in loader:
            if n_samples >= fisher_size:
                break
            x = x.to(device)
            remaining = fisher_size - n_samples
            if x.size(0) > remaining:        # truncate last batch to hit exactly fisher_size
                x = x[:remaining]
            out = self.model(x)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            pseudo_labels = logits.detach().argmax(1)
            loss = ce_loss_fn(logits, pseudo_labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in self.model.named_parameters():
                if n in fisher_acc and p.grad is not None:
                    fisher_acc[n].add_(p.grad.detach().pow(2))
            n_samples += x.size(0)
            n_batches += 1
        if n_batches > 0:
            self._fishers = {n: v / n_batches for n, v in fisher_acc.items()}
        else:
            self._fishers = {n: torch.zeros_like(v) for n, v in fisher_acc.items()}
        self.optimizer.zero_grad(set_to_none=True)

    @torch.enable_grad()
    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out[0] if isinstance(out, (list, tuple)) else out

        entropys = _softmax_entropy(logits)
        ids1 = torch.where(entropys < self.e_margin)[0]
        if not ids1.numel():
            return logits

        entropys_sel = entropys[ids1]
        probs1 = logits[ids1].softmax(1)

        # redundancy filter via cosine similarity
        ids2 = ids1  # fallback: keep all reliable
        if self._current_probs is not None:
            cos = torch.nn.functional.cosine_similarity(
                self._current_probs.unsqueeze(0), probs1, dim=1
            )
            local2 = torch.where(torch.abs(cos) < self.d_margin)[0]
            entropys_sel = entropys_sel[local2]
            probs_sel = probs1[local2]
        else:
            probs_sel = probs1

        self._current_probs = self._ema_probs(self._current_probs, probs_sel.detach())

        if not entropys_sel.numel():
            return logits

        # The first call happens once the model is on its device, so the Fisher diagonal is
        # allocated here. EATA estimates it before adaptation and keeps it fixed.
        if self.fisher_alpha > 0 and self._params0 is None:
            self._params0 = {n: p.detach().clone()
                             for n, p in self.model.named_parameters() if p.requires_grad}
            self._fishers = None  # will be set after first backward

        coeff = 1.0 / torch.exp(entropys_sel.clone().detach() - self.e_margin)
        loss = (entropys_sel * coeff).mean()

        # EWC regularization (only once Fisher is available from previous batch).
        if self.fisher_alpha > 0 and self._fishers is not None:
            ewc = sum(
                self.fisher_alpha * (self._fishers[n] * (p - self._params0[n]) ** 2).sum()
                for n, p in self.model.named_parameters()
                if n in self._fishers
            )
            loss = loss + ewc

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Fisher is estimated from the first batch's entropy gradient and then held fixed: updating
        # it online feeds drifting parameters back into a larger penalty and the stream destabilises.
        if self.fisher_alpha > 0 and self._fishers is None:
            self._fishers = {
                n: p.grad.detach().pow(2)
                for n, p in self.model.named_parameters()
                if n in self._params0 and p.grad is not None
            }
            # Params with no gradient get zero Fisher (no EWC penalty).
            for n in self._params0:
                if n not in self._fishers:
                    self._fishers[n] = torch.zeros_like(self._params0[n])

        self.optimizer.step()

        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.model(x)
        logits = None
        for _ in range(self.steps):
            logits = self._adapt(x)
        return logits
