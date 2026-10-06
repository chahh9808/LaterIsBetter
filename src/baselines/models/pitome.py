# Adapted, with modifications, from https://github.com/hchautran/PiToMe (CC BY-NC 4.0, non-commercial use only).
from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Attention, Block, VisionTransformer


def _scatter_reduce_compat(
    dst: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    reduce: str,
) -> torch.Tensor:
    if hasattr(dst, "scatter_reduce"):
        return dst.scatter_reduce(-2, index, src, reduce=reduce)

    out = dst.clone()
    batch, _, channels = src.shape
    for b in range(batch):
        labels = index[b, :, 0].to(torch.long)
        if reduce in {"sum", "mean"}:
            out[b].index_add_(0, labels, src[b])
            if reduce == "mean":
                counts = torch.zeros(out.shape[1], device=src.device, dtype=src.dtype)
                ones = torch.ones_like(labels, device=src.device, dtype=src.dtype)
                counts.index_add_(0, labels, ones)
                out[b] = out[b] / counts.clamp_min(1.0).unsqueeze(-1)
        elif reduce == "amax":
            for i in range(src.shape[1]):
                out[b, labels[i]] = torch.maximum(out[b, labels[i]], src[b, i])
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
    return out


def _do_nothing(x, mode=None):
    return x


def _pitome(
    metric: torch.Tensor,
    *,
    class_token: bool,
    indices: torch.Tensor,
    scores: torch.Tensor,
    r: int,
) -> Callable[[torch.Tensor, str], torch.Tensor]:
    batch, tokens, _ = scores.shape
    merge_idx = indices[..., : 2 * r]
    protected_idx = indices[..., 2 * r :]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(batch, tokens, r))
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(batch, r, r))
    _, dst_idx = scores.max(dim=-1)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            x_cls = None

        bsz, _, channels = x.shape
        batch_idx = torch.arange(bsz, device=x.device).unsqueeze(1)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]

        if mode != "prune":
            scatter_index = dst_idx.unsqueeze(-1).expand(bsz, r, channels)
            # scatter INTO dst (include self), matching official pitome merge: the
            # destination token retains its own value and accumulates matched sources.
            # (was scatter into zeros, which discarded the dst token -- a large info loss
            # at every merge that compounded over the full val set.)
            dst = _scatter_reduce_compat(dst, scatter_index, src, reduce=mode)

        merged = torch.cat([protected, dst], dim=1)
        return torch.cat([x_cls, merged], dim=1) if x_cls is not None else merged

    return merge


def _pitome_bsm(
    metric: torch.Tensor,
    *,
    class_token: bool,
    r: int,
) -> Callable[[torch.Tensor, str], torch.Tensor]:
    # Plain ToMe-style bipartite soft matching on RAW even/odd token positions.
    # The official PiToMe (hchautran/PiToMe, algo/pitome/merge.py::bsm) applies this
    # in the first ceil(L/2) layers and energy-based pitome in the rest. metric is the
    # L2-normalized patch metric (class token already stripped by pitome_vision).
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[..., r:, :]
    src_idx = edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if class_token:
            x_cls = x[:, :1, :]
            x = x[:, 1:, :]
        else:
            x_cls = None

        src, dst = x[..., ::2, :], x[..., 1::2, :]
        bsz, t1, channels = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(bsz, t1 - r, channels))

        if mode != "prune":
            src = src.gather(dim=-2, index=src_idx.expand(bsz, r, channels))
            scatter_index = dst_idx.expand(bsz, r, channels)
            # scatter INTO dst (include self), matching official bsm: the destination
            # token keeps its own value and accumulates the matched source tokens.
            dst = _scatter_reduce_compat(dst, scatter_index, src, reduce=mode)

        merged = torch.cat([unm, dst], dim=1)
        return torch.cat([x_cls, merged], dim=1) if x_cls is not None else merged

    return merge


def pitome_vision(
    metric: torch.Tensor,
    *,
    ratio: float,
    margin: float,
    class_token: bool,
    use_bsm_pitome: bool,
) -> Callable[[torch.Tensor, str], torch.Tensor]:
    if class_token:
        metric = metric[:, 1:, :]

    batch, tokens, _ = metric.shape
    if ratio >= 1.0:
        return _do_nothing

    r = math.floor(tokens - tokens * ratio)
    if r <= 0:
        return _do_nothing

    with torch.no_grad():
        metric = F.normalize(metric, p=2, dim=-1)
        sim = metric @ metric.transpose(-1, -2)
        energy_score = F.elu(sim - margin, alpha=1.0).mean(dim=-1)
        indices = torch.argsort(energy_score, descending=True)

    if use_bsm_pitome:
        return _pitome_bsm(metric, class_token=class_token, r=r)
    return _pitome(metric, class_token=class_token, indices=indices, scores=sim, r=r)


def merge_source(
    merge: Callable[[torch.Tensor, str], torch.Tensor],
    x: torch.Tensor,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    if source is None:
        batch, tokens, _ = x.shape
        source = torch.eye(tokens, device=x.device)[None, ...].expand(batch, tokens, tokens)
    return merge(source, mode="amax")


def merge_wavg(
    merge: Callable[[torch.Tensor, str], torch.Tensor],
    x: torch.Tensor,
    size: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size.clamp_min(1e-6)
    return x, size


class PiToMeAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)


class PiToMeBlock(Block):
    def init_margin(self, margin: float = 0.5):
        self.margin = float(margin)

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn, metric = self.attn(self.norm1(x))
        x = x + self._drop_path1(x_attn)

        ratio = float(self._info["ratio"].pop(0))
        use_bsm_pitome = bool(self._info["use_bsm_pitome"].pop(0))

        if ratio < 1.0:
            merge = pitome_vision(
                metric=metric,
                ratio=ratio,
                margin=self.margin,
                class_token=self._info["class_token"],
                use_bsm_pitome=use_bsm_pitome,
            )
            if self._info["trace_source"]:
                self._info["source"] = merge_source(merge, x, self._info["source"])
            x, self._info["size"] = merge_wavg(merge, x, self._info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


def _resolve_local_ratios(keep_rate: list[float], reduction_loc: list[int], depth: int, init_tokens: int) -> list[float]:
    if not keep_rate:
        return [1.0] * depth

    if len(keep_rate) == 1:
        rate = float(keep_rate[0])
        token_ratio = [rate for _ in reduction_loc]
    else:
        token_ratio = [float(v) for v in keep_rate]

    if len(token_ratio) != len(reduction_loc):
        raise ValueError(f"Mismatch between reduction locations {reduction_loc} and keep rates {token_ratio}")

    local_ratios = [1.0] * depth
    prev_tokens = int(init_tokens)
    for idx, loc in enumerate(reduction_loc):
        target_tokens = max(1, int(token_ratio[idx] * init_tokens))
        target_tokens = min(prev_tokens, target_tokens)
        local_ratios[loc] = float(target_tokens) / float(max(prev_tokens, 1))
        prev_tokens = target_tokens
    return local_ratios


class PiToMeVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        args=None,
        dyvit_distillation=False,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            representation_size,
            distilled,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            embed_layer,
            norm_layer,
            act_layer,
            weight_init,
        )

        reduction_loc = list(args.reduction_loc)
        self._ratio_schedule = _resolve_local_ratios(args.keep_rate, reduction_loc, depth, self.patch_embed.num_patches)
        self.ratio = 1.0
        self._info = {
            "ratio": [],
            "use_bsm_pitome": [],
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": getattr(args, "proportional_attn", True),
            "class_token": self.cls_token is not None,
            "distill_token": hasattr(self, "dist_token") and self.dist_token is not None,
            "pitome_use_bsm_pitome": bool(getattr(args, "pitome_use_bsm_pitome", False)),
        }
        self._last_effective_gflops = None

        margins = [0.75 - 0.75 * (i / depth) for i in range(depth)]
        for i, block in enumerate(self.blocks):
            block.__class__ = PiToMeBlock
            block.init_margin(margins[i])
            block._info = self._info
            block.attn.__class__ = PiToMeAttention

    def calculate_block_flop(self, shape: Tuple[int, int, int]) -> float:
        _, tokens, channels = shape
        mhsa_flops = 4 * tokens * channels * channels + 2 * tokens * tokens * channels
        ffn_flops = 8 * tokens * channels * channels
        return float(mhsa_flops + ffn_flops)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if getattr(self, "dist_token", None) is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        self.total_flop = 0.0
        for block in self.blocks:
            self.total_flop += self.calculate_block_flop(tuple(x.shape))
            x = block(x)

        x = self.norm(x)
        if getattr(self, "dist_token", None) is None:
            return self.pre_logits(x[:, 0])
        return x[:, 0], x[:, 1]

    def forward(self, x):
        ratio_schedule = list(self._ratio_schedule)
        if len(ratio_schedule) != len(self.blocks):
            ratio_schedule = [1.0] * len(self.blocks)
        self._info["ratio"] = ratio_schedule
        if self._info["pitome_use_bsm_pitome"]:
            num_bsm_layers = math.ceil(len(self.blocks) * 0.5)
            self._info["use_bsm_pitome"] = [True] * num_bsm_layers + [False] * (len(self.blocks) - num_bsm_layers)
        else:
            self._info["use_bsm_pitome"] = [False] * len(self.blocks)
        self._info["size"] = None
        self._info["source"] = None

        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            x = (x + x_dist) / 2 if not self.training else (x, x_dist)
        else:
            x = self.head(x)

        self._last_effective_gflops = float(self.total_flop) / 1e9
        return x
