# Adapted, with modifications, from https://github.com/mr-eggplant/FOA (NTUitive license, non-commercial use only).
"""FOA: Forward Optimization Adaptation (Niu et al., ICML 2024).

Port of the official FOA implementation at https://github.com/mr-eggplant/FOA.

Key elements, as in that implementation:
  - CMA-ES (gradient-free) optimization of prompt embeddings
  - num_prompts = 3 (default in official main.py)
  - fitness_lambda = 0.4 (entropy + feature-statistics MSE)
  - Source statistics (per-layer CLS feature mean/std on clean ImageNet train),
    computed once per (backbone, schedule) and cached as a .pt file.
  - Activation shifting via running CLS mean (Eqn. 7-9 in paper).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cma
import numpy as np
import torch
import torch.nn as nn

from src.utils.entropy import _softmax_entropy
from src.utils.prompt_utils import _get_inner_model


def _patch_tome_to_protect_prefix(n_protected: int) -> None:
    """Monkey-patch ToMe's bipartite_soft_matching to protect the first
    `n_protected` tokens (CLS + prompt tokens) from being merged.

    Replaces the patched function with a re-implementation that masks rows
    (for even-positioned protected tokens, which fall in the "a"/source set)
    and columns (for odd-positioned protected tokens, in the "b"/destination
    set) of the similarity matrix to -inf, so protected tokens never become
    src and no "a" merges into a protected "b".
    """
    import importlib
    tome_patch = importlib.import_module("tome.patch.timm")  # the package attribute of that name is apply_patch itself

    if getattr(tome_patch, "_foa_prefix_patched", False):
        tome_patch._foa_n_protected = max(int(n_protected),
                                          int(getattr(tome_patch, "_foa_n_protected", 0)))
        return

    original = tome_patch.bipartite_soft_matching
    tome_patch._foa_original_bsm = original
    tome_patch._foa_n_protected = int(n_protected)
    tome_patch._foa_prefix_patched = True

    def patched(metric, r, class_token=False, distill_token=False):
        n = int(getattr(tome_patch, "_foa_n_protected", 0))
        if n <= 1:
            return original(metric, r, class_token=class_token,
                            distill_token=distill_token)

        # As in bipartite_soft_matching, r is clamped to the mergeable source pairs: the protected
        # prompt prefix leaves fewer of them than a late schedule's budget asks for.
        t = int(metric.shape[-2])
        r = min(int(r), max(0, (t - n) // 2))
        if r <= 0:
            def _identity_merge(x, mode: str = "mean"):
                return x
            def _identity_unmerge(x):
                return x
            return _identity_merge, _identity_unmerge

        protect_size = int(metric.shape[-1])
        with torch.no_grad():
            metric_norm = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            a, b = metric_norm[..., ::2, :], metric_norm[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            # Protect tokens at positions [0..n-1]:
            #   even index 2k → row k in a (set its row to -inf)
            #   odd index 2k+1 → column k in b (set its column to -inf)
            for k in range(n):
                if k % 2 == 0:
                    scores[..., k // 2, :] = float("-inf")
                else:
                    scores[..., :, k // 2] = float("-inf")
            if distill_token:
                scores[..., :, 0] = float("-inf")
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            B, T1, C = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(B, T1 - r, C))
            src = src.gather(dim=-2, index=src_idx.expand(B, r, C))
            # Deterministic scatter-mean (see tome_extensions.merge for rationale).
            dst_len = dst.shape[-2]
            onehot = torch.nn.functional.one_hot(dst_idx[..., 0], dst_len).to(src.dtype)
            count = onehot.sum(dim=1)
            src_sum = onehot.transpose(1, 2) @ src
            if mode == "mean":
                dst = (dst + src_sum) / (1.0 + count).unsqueeze(-1)
            else:
                dst = dst + src_sum
            if distill_token:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            return x

        return merge, unmerge

    tome_patch.bipartite_soft_matching = patched


def _restore_tome_patch() -> None:
    """Revert the FOA-prefix monkey-patch (if active)."""
    import importlib
    tome_patch = importlib.import_module("tome.patch.timm")  # the package attribute of that name is apply_patch itself
    if getattr(tome_patch, "_foa_prefix_patched", False):
        tome_patch.bipartite_soft_matching = tome_patch._foa_original_bsm
        tome_patch._foa_prefix_patched = False


class PromptViT(nn.Module):
    """ViT wrapper that inserts learnable prompt tokens after CLS, mirroring official FOA."""

    def __init__(self, vit: nn.Module, num_prompts: int = 3):
        super().__init__()
        self.vit = vit
        self.num_prompts = int(num_prompts)
        self.embed_dim = int(getattr(vit, "embed_dim", 384))
        if self.num_prompts > 0:
            patch_size = getattr(vit.patch_embed, "patch_size", (16, 16))
            psz = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
            val = math.sqrt(6.0 / float(3 * psz * psz + self.embed_dim))
            self.prompts = nn.Parameter(
                torch.zeros(1, self.num_prompts, self.embed_dim).uniform_(-val, val)
            )
        else:
            self.prompts = None

    def reset_prompts(self):
        if self.prompts is None:
            return
        patch_size = getattr(self.vit.patch_embed, "patch_size", (16, 16))
        psz = patch_size[0] if isinstance(patch_size, (tuple, list)) else int(patch_size)
        val = math.sqrt(6.0 / float(3 * psz * psz + self.embed_dim))
        with torch.no_grad():
            self.prompts.data.uniform_(-val, val)

    def _inject_prompts(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompts is None:
            return x
        B = x.shape[0]
        p = self.prompts.expand(B, -1, -1).to(x.device).to(x.dtype)
        return torch.cat([x[:, :1], p, x[:, 1:]], dim=1)

    def _embed(self, x: torch.Tensor, inject: bool) -> torch.Tensor:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        if inject:
            x = self._inject_prompts(x)
        if hasattr(self.vit, "norm_pre"):
            x = self.vit.norm_pre(x)
        return x

    def _setup_tome_state(self) -> None:
        """Reset per-forward ToMe state. Mirrors what tome's wrapped
        VisionTransformer.forward does at the start of every forward."""
        tinfo = getattr(self.vit, "_tome_info", None)
        if tinfo is None:
            return
        from tome.utils import parse_r
        tinfo["r"] = parse_r(len(self.vit.blocks), getattr(self.vit, "r", 0))
        tinfo["size"] = None
        tinfo["source"] = None

    def _collect_layer_cls(self, x: torch.Tensor) -> torch.Tensor:
        self._setup_tome_state()
        L = len(self.vit.blocks)
        feats = []
        for i in range(L):
            x = self.vit.blocks[i](x)
            if i < L - 1:
                feats.append(self.vit.blocks[i + 1].norm1(x[:, 0]))
            else:
                feats.append(self.vit.norm(x[:, 0]))
        return torch.cat(feats, dim=1)

    def layers_cls_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-layer CLS feature, NO prompts (used to collect source statistics)."""
        return self._collect_layer_cls(self._embed(x, inject=False))

    def layers_cls_features_with_prompts(self, x: torch.Tensor) -> torch.Tensor:
        """Per-layer CLS feature, WITH prompts (used at test time)."""
        return self._collect_layer_cls(self._embed(x, inject=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embed(x, inject=True)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return self.vit.forward_head(x)


class FOA(nn.Module):
    """Official FOA: CMA-ES optimisation of prompt embeddings with feature-stats fitness.

    Source statistics (per-layer CLS feature mean/std) must be provided via
    `obtain_origin_stat(loader)` before the first TTA batch, or loaded from
    `source_stats_path`. Falls back to entropy-only fitness if unavailable.
    """

    def __init__(
        self,
        model: nn.Module,
        num_prompts: int = 3,
        fitness_lambda: float = 0.4,
        popsize: int | None = None,
        source_stats_path: str = "",
        seed: int = 2020,
    ):
        super().__init__()
        inner = _get_inner_model(model)
        # Wrap if not already wrapped
        if isinstance(model, PromptViT):
            self.pvit = model
        elif isinstance(inner, PromptViT):
            self.pvit = inner
        else:
            self.pvit = PromptViT(inner, num_prompts=num_prompts)

        # Freeze everything
        for p in self.pvit.vit.parameters():
            p.requires_grad_(False)
        if self.pvit.prompts is not None:
            self.pvit.prompts.requires_grad_(False)

        self.fitness_lambda = float(fitness_lambda)
        self.num_prompts = int(num_prompts)
        self.embed_dim = self.pvit.embed_dim
        self.seed = int(seed)
        self.source_stats_path = str(source_stats_path or "")
        self.train_info: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.source_stats_path and Path(self.source_stats_path).exists():
            blob = torch.load(self.source_stats_path, map_location="cpu")
            self.train_info = (blob["std"], blob["mean"])

        self.hist_stat: torch.Tensor | None = None

        if self.pvit.prompts is not None:
            dim = self.pvit.prompts.numel()
            self._init_cma(dim, popsize)
            self.best_prompts = self.pvit.prompts.detach().clone()
        else:
            self.es = None
            self.best_prompts = None
        self.best_loss = float("inf")
        self.best_outputs: torch.Tensor | None = None
        self._mse = nn.MSELoss(reduction="none")

        # Activate ToMe prefix protection if prompts are inserted
        if self.num_prompts > 0:
            _patch_tome_to_protect_prefix(1 + self.num_prompts)

    def _init_cma(self, dim: int, popsize: int | None) -> None:
        if popsize is None:
            popsize = int(4 + 3 * math.log(max(dim, 2)))
        self.es = cma.CMAEvolutionStrategy(
            dim * [0], 1,
            inopts={"seed": self.seed, "popsize": int(popsize), "maxiter": -1, "verbose": -1},
        )

    @torch.no_grad()
    def obtain_origin_stat(self, train_loader, n_batches: int = 50) -> None:
        """Compute per-layer CLS feature mean/std on clean source images and cache."""
        self.pvit.eval()
        device = next(self.pvit.vit.parameters()).device
        feats = []
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            feats.append(self.pvit.layers_cls_features(x).detach())
        feats = torch.cat(feats, dim=0)
        std, mean = torch.std_mean(feats, dim=0)
        self.train_info = (std.detach().cpu(), mean.detach().cpu())
        if self.source_stats_path:
            Path(self.source_stats_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"std": self.train_info[0], "mean": self.train_info[1]},
                       self.source_stats_path)

    def _update_hist(self, batch_mean_last: torch.Tensor) -> None:
        if self.hist_stat is None:
            self.hist_stat = batch_mean_last.detach().clone()
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean_last

    def _shift_vector(self) -> torch.Tensor | None:
        if self.hist_stat is None or self.train_info is None:
            return None
        device = self.hist_stat.device
        src_mean_last = self.train_info[1][-self.embed_dim:].to(device)
        return src_mean_last - self.hist_stat

    @torch.no_grad()
    def _forward_loss(self, x: torch.Tensor, shift: torch.Tensor | None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.pvit.layers_cls_features_with_prompts(x)
        B = x.shape[0]
        batch_std, batch_mean = torch.std_mean(feats, dim=0)
        if self.train_info is None:
            disc = torch.zeros((), device=feats.device)
        else:
            sm = self.train_info[1].to(feats.device)
            ss = self.train_info[0].to(feats.device)
            disc = self.fitness_lambda * (
                self._mse(batch_std, ss).sum() + self._mse(batch_mean, sm).sum()
            ) * B / 64.0
        cls_feat = feats[:, -self.embed_dim:]
        head = self.pvit.vit.head
        out = head(cls_feat)
        ent = _softmax_entropy(out).sum()
        loss = disc + ent
        if shift is not None:
            out = head(cls_feat + shift.to(cls_feat.device))
        return out, loss, batch_mean

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pvit.prompts is None or self.es is None:
            return self.pvit(x)
        shift = self._shift_vector()
        self.best_loss = float("inf")
        self.best_outputs = None
        batch_means_last: list[torch.Tensor] = []
        device = next(self.pvit.vit.parameters()).device
        prompt_shape = self.pvit.prompts.shape

        candidates = list(self.es.ask()) + [self.best_prompts.flatten().detach().cpu().numpy()]
        losses = []
        for prompt_flat in candidates:
            p = torch.tensor(prompt_flat, dtype=torch.float, device=device).reshape(prompt_shape)
            self.pvit.prompts = nn.Parameter(p, requires_grad=False)
            out, loss, batch_mean = self._forward_loss(x, shift)
            batch_means_last.append(batch_mean[-self.embed_dim:].unsqueeze(0))
            if loss.item() < self.best_loss:
                self.best_loss = float(loss.item())
                self.best_prompts = self.pvit.prompts.detach().clone()
                self.best_outputs = out
            losses.append(float(loss.item()))
        # Restore best prompts as current
        self.pvit.prompts = nn.Parameter(self.best_prompts.detach().clone(), requires_grad=False)
        self.es.tell(candidates, losses)
        self._update_hist(torch.cat(batch_means_last, dim=0).mean(0))
        return self.best_outputs

    def reset(self):
        if self.pvit.prompts is not None and self.es is not None:
            dim = self.pvit.prompts.numel()
            self._init_cma(dim, None)
            self.pvit.reset_prompts()
            self.best_prompts = self.pvit.prompts.detach().clone()
        self.hist_stat = None
        self.best_loss = float("inf")
