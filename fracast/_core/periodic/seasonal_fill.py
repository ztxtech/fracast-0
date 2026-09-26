"""Seasonal fill and period detection utilities.

These functions provide a parameter-free seasonal prior for the quantile
decoder. They follow the reference ``_seasonal_naive`` implementation and are
intended to be paired with a causal convolution that propagates local shape
across the forecast horizon.

Reference:
- ``raws-labs/tinycast``, ``tinycast/backbone.py``, ``_seasonal_naive``.
- Period detection uses ``official_periodogram.significant_periods``.

Two adaptations are intentional:

1. Masked inputs are supported. Phase means are computed only from valid
   observations.
2. Empty phase bins fall back to the mean of valid observations in the row.
   With no missing values this is equivalent to the reference behavior.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fracast._core.periodic.encoder import _fill_nan_row
from fracast._core.periodic.official_periodogram import significant_periods


@torch.compiler.disable()
def detect_periods(x: torch.Tensor, mask: torch.Tensor | None = None, *,
                   top_k: int = 4, min_period: int = 2,
                   alpha: float = 0.05) -> torch.Tensor:
    """Return the top-k significant periods for each row.

    Non-finite values are replaced by the row mean before detection. When no
    period is significant, the corresponding slot remains zero and downstream
    code applies a ``clamp(min=1)`` guard.
    """
    _batch, length = x.shape
    values = _fill_nan_row(x.float(), mask)
    periods, _scores, _n_valid = significant_periods(
        values,
        min_period=int(min_period),
        max_period=max(2, length // 2),
        top_k=int(top_k),
        significance_alpha=float(alpha),
    )
    return periods


def folded_seasonal_fill(x: torch.Tensor, fut_pos: torch.Tensor,
                         periods: torch.Tensor, phase_bins: int = 16,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
    """Fold the context by phase and retrieve values for future positions.

    ``x`` has shape ``[B, L]`` and must use the model's normalization space.
    ``fut_pos`` has shape ``[B, H]`` and contains absolute forecast positions.
    ``periods`` has shape ``[B, K]``; column zero is the primary period.
    The returned tensor has shape ``[B, H]``.
    """
    if x.dim() != 2:
        raise ValueError(
            f"folded_seasonal_fill expects [B, L], received {tuple(x.shape)}")
    batch, length = x.shape
    n_bins = int(phase_bins)
    if n_bins < 1:
        raise ValueError(f"phase_bins must be at least 1, received {n_bins}")
    dtype = x.dtype
    weights = torch.ones_like(x) if mask is None else mask.to(dtype)
    values = torch.where(
        weights > 0,
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x),
    )

    positions = torch.arange(length, device=x.device).view(1, length).float()
    primary = periods[:, :1].clamp(min=1).float()
    phase_index = torch.clamp(
        ((positions % primary) / primary * n_bins).long(), max=n_bins - 1)
    one_hot = F.one_hot(phase_index, n_bins).to(dtype) * weights.unsqueeze(-1)
    counts = one_hot.sum(dim=1)
    totals = torch.bmm(
        one_hot.transpose(1, 2), values.unsqueeze(-1)).squeeze(-1)
    global_mean = (
        values.sum(dim=1, keepdim=True)
        / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    )
    phase_mean = torch.where(
        counts > 0,
        totals / counts.clamp(min=1.0),
        global_mean.expand(batch, n_bins),
    )
    future_phase = torch.clamp(
        (fut_pos.float() % primary) / primary * n_bins,
        max=n_bins - 1,
    ).long()
    return torch.gather(phase_mean, 1, future_phase)


def last_period_fill(x: torch.Tensor, fut_pos: torch.Tensor,
                     periods: torch.Tensor, phase_bins: int = 16,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """Copy values from the most recent complete period.

    For each future position, this selects the closest source in the final part
    of the context with the same phase. Invalid source positions fall back to
    the row mean. ``phase_bins`` is retained for interface compatibility with
    ``folded_seasonal_fill``.
    """
    if x.dim() != 2:
        raise ValueError(
            f"last_period_fill expects [B, L], received {tuple(x.shape)}")
    batch, length = x.shape
    dtype = x.dtype
    weights = torch.ones_like(x) if mask is None else mask.to(dtype)
    values = torch.where(
        weights > 0,
        torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        torch.zeros_like(x),
    )

    primary = periods[:, :1].clamp(min=1).long()
    distance = fut_pos.long() - (length - 1)
    source = fut_pos.long() - ((distance + primary - 1) // primary) * primary
    in_range = (source >= 0) & (source < length)
    safe_source = source.clamp(min=0, max=length - 1)
    source_values = torch.gather(values, 1, safe_source)
    valid = in_range
    if mask is not None:
        valid = valid & torch.gather(mask.bool(), 1, safe_source)
    global_mean = (
        values.sum(dim=1, keepdim=True)
        / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    )
    return torch.where(
        valid,
        source_values,
        global_mean.expand(batch, source_values.shape[1]),
    ).to(dtype)
