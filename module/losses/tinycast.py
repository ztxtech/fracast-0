"""TinyCast training objectives used by the FracCast pretraining recipe.

The two functions below are adapted from the Apache-2.0 TinyCast reference
implementation:

    https://github.com/raws-labs/tinycast
    tinycast/losses.py
    tinycast/scale.py

They are kept local so that the pretraining path has no runtime dependency on a
vendored copy of the upstream repository.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch


BASE_SEASONALITY = 24.0
COMMIT_WEIGHT = 0.3


def _as_bh(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 3 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.dim() != 2:
        raise ValueError(f"{name} must be (B, H) or (B, H, 1), got {tuple(x.shape)}")
    return x


def _mask_like(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(ref)
    mask = _as_bh(mask, "mask").to(ref.dtype)
    if mask.shape != ref.shape:
        raise ValueError(f"mask {tuple(mask.shape)} != target {tuple(ref.shape)}")
    return mask


def seasonal_scale_factor(freq: str, domain: Optional[str] = None) -> float:
    """Map a pandas frequency string to TinyCast's seasonal scale factor."""
    has_weekly = domain in {"Transport", "Healthcare", "Sales"}
    if freq == "4S":
        factor = BASE_SEASONALITY / (3600.0 / 4)
    elif freq == "10S":
        factor = BASE_SEASONALITY / 360
    elif freq == "T":
        factor = BASE_SEASONALITY / (24.0 * 60)
    elif freq.endswith("T"):
        factor = BASE_SEASONALITY / (24 * 60 / int(freq[:-1]))
    elif freq == "H":
        factor = BASE_SEASONALITY / 24
    elif freq == "6H":
        factor = BASE_SEASONALITY / 4
    elif freq == "D":
        factor = BASE_SEASONALITY / (7 if has_weekly else 365)
    elif freq.endswith("D") and "WED" not in freq:
        factor = BASE_SEASONALITY / (7 if has_weekly else 365)
        factor *= int(freq[:-1])
    elif freq == "W" or "W-" in freq:
        factor = BASE_SEASONALITY / (365.0 / 7)
    elif freq == "M" or "M-" in freq or freq == "MS":
        factor = BASE_SEASONALITY / 12
    elif "Q" in freq:
        factor = BASE_SEASONALITY / 4.0
    elif "A" in freq:
        factor = BASE_SEASONALITY / 4.0
    else:
        raise NotImplementedError(f"seasonal scale is not implemented for {freq!r}")
    return factor


def seasonal_copy_baseline(
    context: torch.Tensor,
    horizon: int,
    scale_factor: Union[torch.Tensor, float],
    *,
    base_seasonality: float = BASE_SEASONALITY,
) -> torch.Tensor:
    """Repeat the last seasonal cycle of each context over the forecast horizon."""
    if context.dim() == 3 and context.shape[-1] == 1:
        context = context.squeeze(-1)
    if context.dim() != 2:
        raise ValueError(f"context must be (B, L) or (B, L, 1), got {tuple(context.shape)}")
    batch, length = context.shape
    horizon = int(horizon)
    if horizon < 1 or length < 4:
        raise ValueError(f"invalid horizon/length: H={horizon}, L={length}")

    sf = torch.as_tensor(scale_factor, dtype=torch.float32, device=context.device)
    sf = sf.reshape(-1)
    if sf.numel() == 1:
        sf = sf.expand(batch)
    elif sf.numel() != batch:
        raise ValueError(f"scale_factor has {sf.numel()} entries for batch={batch}")

    lag = (base_seasonality / sf.clamp(min=1e-3)).round().long()
    lag = lag.clamp(2, max(2, length // 2)).view(-1, 1)
    h = torch.arange(horizon, device=context.device).view(1, horizon)
    src = (length - lag + (h % lag)).clamp(0, length - 1)
    return torch.gather(context, 1, src)


def committing_loss(
    median: torch.Tensor,
    target: torch.Tensor,
    seasonal_copy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    weight: float = COMMIT_WEIGHT,
    gated: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gated hinge that encourages the median to be at least as good as the copy."""
    median = _as_bh(median, "median")
    target = _as_bh(target, "target")
    seasonal_copy = _as_bh(seasonal_copy, "seasonal_copy")
    if median.shape != target.shape or median.shape != seasonal_copy.shape:
        raise ValueError(
            "median, target and seasonal_copy must have the same (B, H) shape"
        )

    obs = _mask_like(mask, target)
    med_err = (median - target).abs()
    copy_err = (seasonal_copy - target).abs()
    finite = (
        torch.isfinite(med_err)
        & torch.isfinite(copy_err)
        & torch.isfinite(target)
    ).to(obs.dtype)
    scored = obs * finite

    hinge = torch.relu(med_err - copy_err)
    hinge = torch.where(torch.isfinite(hinge), hinge, torch.zeros_like(hinge))
    per_sample = (hinge * scored).sum(dim=1) / obs.sum(dim=1).clamp(min=1.0)
    if gated:
        med_total = (torch.nan_to_num(med_err, 0.0, 0.0, 0.0) * scored).sum(dim=1)
        copy_total = (torch.nan_to_num(copy_err, 0.0, 0.0, 0.0) * scored).sum(dim=1)
        per_sample = per_sample * (copy_total < med_total).to(per_sample.dtype)
    per_sample = per_sample * float(weight)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unknown reduction: {reduction!r}")


__all__ = [
    "BASE_SEASONALITY",
    "COMMIT_WEIGHT",
    "committing_loss",
    "seasonal_copy_baseline",
    "seasonal_scale_factor",
]
