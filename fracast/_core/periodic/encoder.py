"""Periodogram-based phase encoding for multi-resolution inputs.

The algorithm is adapted from the Apache-2.0 TinyCast implementation:

- ``tinycast/periodogram.py::significant_periods`` is copied verbatim to
  ``official_periodogram.py``.
- ``tinycast/encoding.py::_phase_encoding`` is copied verbatim to
  ``official_encoding.py``.

Fracast extends those primitives in three ways:

1. Period detection runs independently for each pyramid level. Positions are
   expressed in that level's stride so periods remain local to the level.
2. The bounded-recency basis is not used here. Time position is handled by the
   model's existing temporal encoding.
3. The reference period-trust gate and folded seasonal profile are not part of
   this encoder. They are separate mechanisms and can be evaluated later.

The detector itself is unchanged. This includes the Bonferroni threshold,
local-max filtering, top-k semantics, and rounding rather than truncation.

Input values may contain NaN. Before detection, non-finite values are replaced
by the mean of valid values in the same row. If a row has no valid values, its
missing entries are replaced by zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fracast._core.periodic.official_encoding import _phase_encoding
from fracast._core.periodic.official_periodogram import significant_periods


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _fill_nan_row(v: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
    """Replace non-finite values with the row mean of valid observations."""
    v = torch.nan_to_num(v, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
    finite = torch.isfinite(v)
    if m is not None:
        finite = finite & m
    count = finite.sum(dim=1, keepdim=True).clamp(min=1)
    total = torch.where(finite, v, torch.zeros_like(v)).sum(dim=1, keepdim=True)
    mean = total / count
    return torch.where(finite, v, mean)


class PeriodicPhaseEncoder(nn.Module):
    """Detect significant periods and project their phase encoding.

    The detector has no learnable parameters. A single linear projection maps
    the phase features to the model width. Rejected period slots are encoded as
    zeros, matching the reference semantics.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg, pyramid_cfg = cfg["model"], cfg["pyramid"]
        self.d = int(model_cfg["d_model"])
        self.W = int(model_cfg["W"])
        self.top_k = int(model_cfg.get("periodic_topk", 16))
        self.n_harm = int(model_cfg.get("periodic_harmonics", 1))
        self.min_period = int(model_cfg.get("periodic_min_period", 2))
        self.alpha = float(model_cfg.get("periodic_alpha", 0.05))

        ratios = list(pyramid_cfg["ratios"])
        self.L = len(ratios) + 1
        strides = []
        for row in range(self.L):
            level = self.L - 1 - row
            stride = 1
            for ratio in ratios[:level]:
                stride *= int(ratio)
            strides.append(stride)
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long),
                             persistent=False)

        n_bins = _next_pow2(max(2, self.W)) // 2
        self.K = min(self.top_k, n_bins)
        self.proj = nn.Linear(2 * self.K * self.n_harm, self.d)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None,
                widths) -> torch.Tensor:
        """Encode flattened coarse-to-fine pyramid inputs ``[B, N]``."""
        batch, n_tokens = values.shape
        widths = [int(w) for w in widths]
        if len(widths) != self.L:
            raise ValueError(
                f"PeriodicPhaseEncoder expects L={self.L} levels, received {len(widths)}")
        if sum(widths) != n_tokens:
            raise ValueError(
                f"Width sum {sum(widths)} does not match token count {n_tokens}")
        if max(widths) != self.W:
            raise ValueError(
                f"Finest level width {max(widths)} must equal model.W={self.W}")

        device = values.device
        features = values.new_zeros((batch, n_tokens, 2 * self.K * self.n_harm))
        offset = 0
        for row, width in enumerate(widths):
            level_values = values[:, offset:offset + width].float()
            level_mask = mask[:, offset:offset + width].bool() if mask is not None else None
            level_values = _fill_nan_row(level_values, level_mask)
            positions = torch.arange(width, device=device, dtype=torch.long)
            positions = positions.unsqueeze(0).expand(batch, width)
            # Period detection is parameter-free and returns integer periods.
            with torch.no_grad():
                periods, _scores, _n_valid = significant_periods(
                    level_values,
                    min_period=self.min_period,
                    max_period=max(2, width // 2),
                    top_k=self.top_k,
                    significance_alpha=self.alpha,
                )
            if periods.shape[1] != self.K:
                raise ValueError(
                    f"Detector returned K={periods.shape[1]}, expected {self.K}")
            features[:, offset:offset + width] = _phase_encoding(
                positions, periods, n_harmonics=self.n_harm)
            offset += width
        return self.proj(features)
