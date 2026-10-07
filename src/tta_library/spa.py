# Adapted, with modifications, from https://github.com/mr-eggplant/SPA (NTUitive license, non-commercial use only).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.param_utils import _collect_ln_params, _configure_norm_only, _find_classifier


class _LowFrequencyAmplitudeMask(nn.Module):
    """Bernoulli mask on the low-frequency amplitudes, phase untouched (SPA's ``Prompt.masking``).

    The centre ``prompt_alpha`` fraction of the shifted amplitude spectrum is the low band; each of
    its bins survives with probability ``keep_prob``, the high band is kept whole, and the mask is
    made conjugate-symmetric so the inverse transform stays real.
    """

    def __init__(self, keep_prob: float = 0.8, prompt_alpha: float = 0.2):
        super().__init__()
        self.keep_prob = float(keep_prob)
        self.prompt_alpha = float(prompt_alpha)
        self._low_mask: torch.Tensor | None = None

    def _low_band(self, h: int, w: int, device, dtype) -> torch.Tensor:
        if self._low_mask is not None and tuple(self._low_mask.shape[-2:]) == (h, w):
            return self._low_mask.to(device=device, dtype=dtype)
        band_h = max(1, int(h * self.prompt_alpha))
        band_w = max(1, int(w * self.prompt_alpha))
        pad_h = (h - band_h) // 2
        pad_w = (w - band_w) // 2
        core = torch.ones(1, 1, band_h, band_w, device=device, dtype=dtype)
        self._low_mask = F.pad(core, [w - pad_w - band_w, pad_w, h - pad_h - band_h, pad_h]).contiguous()
        return self._low_mask

    @staticmethod
    def _symmetrize(mask: torch.Tensor) -> torch.Tensor:
        """Make the mask conjugate-symmetric about the spectrum centre, as the official code does.

        For an even side the first row/column has no partner, so it is left out; the rest is kept
        only where its point reflection is also kept, and the half below the centre row is forced
        to one so a bin and its mirror are never dropped independently.
        """
        start_h = 1 - (mask.shape[-2] % 2)
        start_w = 1 - (mask.shape[-1] % 2)
        sub = mask[:, :, start_h:, start_w:].clone()
        half_h, half_w = sub.shape[-2] // 2, sub.shape[-1] // 2
        sub[:, :, -half_h:, :] = 1
        sub[:, :, half_h, :half_w] = 1
        sub = sub * torch.flip(sub, dims=[-1, -2])
        mask = mask.clone()
        mask[:, :, start_h:, start_w:] = sub
        return mask

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        fft = torch.fft.fft2(x, dim=(-2, -1))
        amplitude, phase = torch.fft.fftshift(torch.abs(fft)), torch.angle(fft)

        low_band = self._low_band(h, w, x.device, x.dtype)
        keep = torch.empty_like(amplitude).bernoulli_(self.keep_prob)
        mask = self._symmetrize(low_band * keep + (1.0 - low_band))

        amplitude = torch.fft.ifftshift(amplitude * mask)
        spectrum = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        return torch.fft.ifft2(spectrum, dim=(-2, -1), s=[h, w]).real


class _HighFrequencyNoise(nn.Module):
    """Per-patch Gaussian noise injection with learned amplitudes (SPA's ``LearnablePatchErasing``).

    Patch ``i`` is mixed as ``(1 - a_i^2) x + a_i^2 v``. The amplitudes ``a`` are the only
    parameters SPA trains adversarially, and ``rescale_to_budget`` holds their average at
    ``noise_ratio`` so the views stay comparably hard over the stream.
    """

    def __init__(self, noise_ratio: float = 0.4, patch_size: int = 16, img_size: int = 224):
        super().__init__()
        self.noise_ratio = float(noise_ratio)
        self.patch_size = int(patch_size)
        self.img_size = int(img_size)
        grid = self.img_size // self.patch_size
        self.alpha = nn.Parameter(torch.full((grid * grid,), self.noise_ratio ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != (self.img_size, self.img_size):
            raise ValueError(
                f"SPA's noise amplitudes are sized for {self.img_size}x{self.img_size} inputs, got {h}x{w}"
            )
        p = self.patch_size
        gh, gw = h // p, w // p
        # One noise draw per patch, shared across the batch, as in the official implementation.
        noise = torch.randn(1, c, gh, p, gw, p, device=x.device, dtype=x.dtype)
        weight = (self.alpha ** 2).view(1, 1, gh, 1, gw, 1)
        tiles = x.view(b, c, gh, p, gw, p)
        view = (1.0 - weight) * tiles + weight * noise
        # The official bound: the ViT backbones of this table normalise with 0.5/0.5, so a test
        # image lives in [-1, 1] and the noisy view is held to the same range.
        return view.reshape(b, c, h, w).clamp(-1.0, 1.0)

    @torch.no_grad()
    def rescale_to_budget(self) -> None:
        """Renormalise the amplitudes to the fixed average noise fraction (SPA's ``fix_patch_alpha``)."""
        mean = (self.alpha ** 2).mean()
        self.alpha /= torch.sqrt(mean / self.noise_ratio)
        self.alpha.clamp_(-1.0, 1.0)


class SPA(nn.Module):
    """SPA: self-bootstrapping over two geometric-invariant views (Niu et al., ICML 2025).
    Official repo: https://github.com/mr-eggplant/SPA

    Each test batch is predicted once through the model and twice more through a predictor head,
    on a low-frequency amplitude mask and a high-frequency noise view. Where the plain prediction
    is the more confident of the two, it becomes the target of the view's, and the norm-layer
    affine parameters plus the predictor are updated on that KL. The noise amplitudes ascend the
    same loss, which keeps the view as hard as the budget allows. Both augmentations preserve the
    image's geometry, so no pseudo-label or teacher copy is needed.
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 1,
        lr: float = 1e-2,
        momentum: float = 0.9,
        noise_ratio: float = 0.4,
        freq_mask_ratio: float = 0.2,
        predictor_lr_mult: float = 5.0,
    ):
        super().__init__()
        self.model = model
        self.steps = max(1, int(steps))

        params = _configure_norm_only(model, _collect_ln_params)

        # Identity-initialised probe between the pooled feature and the classifier, so the first
        # view forward matches the plain one and the consistency loss starts from no disagreement.
        classifier = _find_classifier(model)
        self.predictor = nn.Linear(classifier.in_features, classifier.in_features, bias=False)
        nn.init.eye_(self.predictor.weight)
        self.predictor.requires_grad_(True)
        self._use_predictor = False
        classifier.register_forward_pre_hook(self._splice_predictor)

        self.masked_view = _LowFrequencyAmplitudeMask(keep_prob=1.0 - float(freq_mask_ratio))
        self.noisy_view = _HighFrequencyNoise(noise_ratio=float(noise_ratio))

        self.optimizer = (
            torch.optim.SGD(
                [
                    {"params": params, "lr": lr},
                    {"params": list(self.predictor.parameters()), "lr": lr * float(predictor_lr_mult)},
                ],
                momentum=momentum,
            )
            if params
            else None
        )
        # The amplitudes take a step of 1, the value the official implementation fixes.
        self.noise_optimizer = torch.optim.SGD([self.noisy_view.alpha], lr=1.0, momentum=momentum)

    def _splice_predictor(self, module: nn.Module, inputs: tuple):
        """Send the pooled feature through the predictor while a view forward is running.

        Hooking the classifier's input keeps the view on the model's own forward, which is what
        resets the token-merging schedule per call; calling forward_features directly would not.
        """
        if not self._use_predictor:
            return None
        return (self.predictor(inputs[0]),)

    def _logits(self, x: torch.Tensor, use_predictor: bool = False) -> torch.Tensor:
        self._use_predictor = use_predictor
        try:
            out = self.model(x)
        finally:
            self._use_predictor = False
        return out[0] if isinstance(out, (list, tuple)) else out

    def _consistency_loss(self, view: torch.Tensor, logits: torch.Tensor):
        """KL from the view's prediction to the plain one, on the samples where the plain one leads."""
        view_logits = self._logits(view, use_predictor=True)
        ids = torch.where(logits.softmax(-1).amax(-1) > view_logits.softmax(-1).amax(-1))[0]
        if not ids.numel():
            return None
        return F.kl_div(
            view_logits[ids].log_softmax(-1), logits[ids].softmax(-1), reduction="batchmean"
        )

    @torch.enable_grad()
    def _adapt(self, x: torch.Tensor) -> torch.Tensor:
        # The plain logits are only a detached target and are what the run reports, so this
        # forward needs no graph: SPA's loss reaches the parameters through the two views.
        with torch.no_grad():
            logits = self._logits(x)

        self.optimizer.zero_grad(set_to_none=True)
        self.noise_optimizer.zero_grad(set_to_none=True)
        adapted = False
        for view in (self.masked_view, self.noisy_view):
            loss = self._consistency_loss(view(x), logits)
            if loss is None:
                continue
            # One backward per view instead of one over their sum: the gradients accumulate to the
            # same values, and only one view's graph is alive at a time, which is what keeps ViT-B
            # at batch 64 within a single card.
            loss.backward()
            adapted = True
        if not adapted:
            return logits

        # The amplitudes maximise what the model minimises.
        if self.noisy_view.alpha.grad is not None:
            self.noisy_view.alpha.grad.neg_()
        self.optimizer.step()
        self.noise_optimizer.step()
        self.noisy_view.rescale_to_budget()

        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.optimizer is None:
            return self.model(x)
        logits = None
        for _ in range(self.steps):
            logits = self._adapt(x)
        return logits
