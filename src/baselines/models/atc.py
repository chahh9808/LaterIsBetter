# Adapted, with modifications, from https://github.com/JoakimHaurum/ATC (MIT License, Copyright (c) 2024 Joakim Bruslund Haurum).
from __future__ import annotations

import ctypes
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from timm.models.layers import DropPath, Mlp, PatchEmbed
from timm.models.vision_transformer import VisionTransformer


def _preload_cuda11_runtime() -> None:
    conda_prefix = Path(torch.__file__).resolve().parents[2]
    nvidia_root = conda_prefix / "site-packages" / "nvidia"
    for lib_path in (
        nvidia_root / "cuda_runtime" / "lib" / "libcudart.so.11.0",
        nvidia_root / "cuda_nvrtc" / "lib" / "libnvrtc.so.11.2",
    ):
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_cuda11_runtime()

try:
    import cupy as cp
    from cuml import AgglomerativeClustering as CuAgglomerativeClustering
except ImportError:
    cp = None
    CuAgglomerativeClustering = None

_logger = logging.getLogger(__name__)


class Attention_ATC(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        metric = k.mean(1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, metric


def _make_agglomerative(num_clusters: int, linkage: str, backend: str = "sklearn"):
    backend = backend.lower()
    if backend == "rapids":
        if CuAgglomerativeClustering is None or cp is None:
            raise RuntimeError("RAPIDS backend requested but cuML/CuPy is not installed.")
        if linkage != "single":
            raise ValueError("RAPIDS ATC backend currently requires linkage='single'.")
        return CuAgglomerativeClustering(
            n_clusters=num_clusters,
            affinity="cosine",
            connectivity="pairwise",
            output_type="numpy",
        )

    try:
        return AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=None,
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            linkage=linkage,
            distance_threshold=None,
        )


def agglomerative_clustering(
    metric: torch.Tensor,
    num_clusters: int,
    linkage: str = "average",
    backend: str = "sklearn",
    clustering=None,
    class_token: bool = False,
    distill_token: bool = False,
):
    protected = int(class_token) + int(distill_token)
    bsz, num_tokens, _ = metric.shape
    num_clusters = min(num_clusters, num_tokens - protected)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = metric @ metric.transpose(-1, -2)

        if class_token:
            scores = scores[:, 1:, 1:]
            num_tokens -= 1

        if backend.lower() == "rapids":
            clustering = clustering or _make_agglomerative(num_clusters=num_clusters, linkage=linkage, backend=backend)
            features = metric.float().contiguous()
            if class_token:
                features = features[:, 1:, :]
            cluster_labels = np.zeros((bsz, num_tokens), dtype=np.int64)
            for batch_idx in range(bsz):
                labels = clustering.fit(features[batch_idx].detach()).labels_
                cluster_labels[batch_idx] = labels
            cluster_labels = torch.from_numpy(cluster_labels).to(device=metric.device)
        else:
            distances = (1.0 - scores).cpu().numpy()
            clustering = _make_agglomerative(num_clusters=num_clusters, linkage=linkage, backend=backend)
            cluster_labels = np.zeros((bsz, num_tokens), dtype=np.int64)
            for batch_idx in range(bsz):
                cluster_labels[batch_idx] = clustering.fit(distances[batch_idx]).labels_
            cluster_labels = torch.from_numpy(cluster_labels).to(device=metric.device)

        if class_token:
            cluster_labels = cluster_labels + protected
            cluster_labels = torch.cat(
                [torch.zeros(bsz, 1, device=metric.device, dtype=torch.long), cluster_labels], dim=-1
            )

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        channels = x.shape[-1]
        dst = torch.zeros(bsz, num_clusters + protected, channels, device=x.device, dtype=x.dtype)
        scatter_index = cluster_labels.unsqueeze(-1).repeat(1, 1, channels)
        if hasattr(dst, "scatter_reduce"):
            return dst.scatter_reduce(-2, scatter_index, x, reduce=mode)

        # Compatibility fallback for older torch builds without Tensor.scatter_reduce.
        out = dst.clone()
        for batch_idx in range(bsz):
            labels = cluster_labels[batch_idx].to(torch.long)
            out[batch_idx].index_add_(0, labels, x[batch_idx])
            if mode == "mean":
                counts = torch.zeros(num_clusters + protected, device=x.device, dtype=x.dtype)
                ones = torch.ones_like(labels, device=x.device, dtype=x.dtype)
                counts.index_add_(0, labels, ones)
                out[batch_idx] = out[batch_idx] / counts.clamp_min(1.0).unsqueeze(-1)
        return out

    return merge, None, cluster_labels


def merge_wavg(merge, x: torch.Tensor, size: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size.clamp_min(1e-6)
    return x, size


class Block_ATC(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_clusters=0,
        linkage="average",
        cls_token=True,
        dist_token=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_ATC(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.num_clusters = num_clusters
        self.linkage = linkage
        self.cluster_backend = "sklearn"
        self.clusterer = None
        self.cls_token = cls_token
        self.dist_token = dist_token

    def forward(self, x, attn_size=None):
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self.drop_path(x_attn)

        cluster_assignment = None
        if self.num_clusters > 0:
            merge, _, cluster_assignment = agglomerative_clustering(
                metric,
                self.num_clusters,
                self.linkage,
                self.cluster_backend,
                self.clusterer,
                self.cls_token,
                self.dist_token,
            )
            x, attn_size = merge_wavg(merge, x, attn_size)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_size, cluster_assignment


class ATCVisionTransformer(VisionTransformer):
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
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            weight_init=weight_init,
        )

        token_ratio = list(args.reduction_ratio)
        reduction_loc = list(args.reduction_loc)
        linkage = args.linkage
        cluster_backend = getattr(args, "cluster_backend", "sklearn")

        if len(token_ratio) == 1:
            token_ratio = [int(self.patch_embed.num_patches * token_ratio[0] ** (idx + 1)) for idx in range(len(reduction_loc))]
        assert len(token_ratio) == len(reduction_loc), f"Mismatch between reduction locations ({reduction_loc}) and token ratios ({token_ratio})"
        print(token_ratio, reduction_loc)

        token_ratio_full = [0 for _ in range(depth)]
        for idx, loc in enumerate(reduction_loc):
            token_ratio_full[loc] = int(token_ratio[idx])

        del self.blocks
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block_ATC(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_clusters=token_ratio_full[i],
                    linkage=linkage,
                )
                for i in range(depth)
            ]
        )
        for block in self.blocks:
            block.cluster_backend = cluster_backend

        self.reduction_loc = reduction_loc
        self.token_ratio = token_ratio
        self.prop_attn = getattr(args, "proportional_attn", True)
        self.viz_mode = getattr(args, "viz_mode", False)
        self.apply(self._init_weights)

    def get_new_module_names(self):
        return []

    def get_reduction_count(self):
        return self.reduction_loc

    def forward(self, x):
        attn_size = None
        bsz = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.viz_mode:
            assignments = {}

        for i, blk in enumerate(self.blocks):
            x, attn_size, cluster_assign = blk(x, attn_size)
            if self.viz_mode and i in self.reduction_loc:
                assignments[i] = cluster_assign.detach().cpu().numpy()
            if not self.prop_attn:
                attn_size = None

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        if self.viz_mode:
            return x, {"Assignment_Maps": assignments}
        return x


def warm_atc_rapids_clusterers(model: nn.Module) -> None:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return
    for block in blocks:
        if getattr(block, "cluster_backend", "").lower() != "rapids":
            continue
        if getattr(block, "num_clusters", 0) <= 0:
            continue
        if getattr(block, "clusterer", None) is None:
            block.clusterer = _make_agglomerative(
                num_clusters=block.num_clusters,
                linkage=block.linkage,
                backend=block.cluster_backend,
            )
