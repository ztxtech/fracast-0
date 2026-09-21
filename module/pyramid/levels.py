"""Pyramid construction for autoregressive rollout during training.

The implementation matches the NumPy data pipeline: each level is produced by
masked mean pooling at the configured ratio, the final ``width`` tokens are
retained, and shorter levels are left-padded with zeros. Values are padded with
zero, masks with false, and coverage with zero. If fewer levels are available
than requested, the finest level is repeated.

This utility is needed because rollout rebuilds the representation for every
context block. The data pipeline constructs the pyramid once for the initial
block; subsequent blocks are constructed on the accelerator here.
"""
from __future__ import annotations

import torch


def resolve_level_widths(width, n_levels: int) -> list[int]:
    """Return one token width per pyramid level, ordered coarse to fine.

    A scalar ``width`` applies the same width to every level. A sequence must
    contain exactly ``n_levels`` positive integers.
    """
    n = int(n_levels)
    if isinstance(width, (list, tuple)):
        widths = [int(w) for w in width]
        if len(widths) != n:
            raise ValueError(
                f"model.level_widths requires {n} values, received {len(widths)}")
        if any(w < 1 for w in widths):
            raise ValueError(f"model.level_widths must be positive, received {widths}")
        return widths
    return [int(width)] * n


def level_span(widths, ratios) -> int:
    """Return the number of source steps covered by the pyramid.

    The span is the maximum of each level width multiplied by that level's
    stride. For uniform widths this equals ``W * product(ratios)``.
    """
    span = int(widths[-1])
    for j in range(len(widths) - 1):
        cumulative = 1
        for ratio in list(ratios)[:len(widths) - 1 - j]:
            cumulative *= int(ratio)
        span = max(span, int(widths[j]) * cumulative)
    return span


def _take_tail(v: torch.Tensor, m: torch.Tensor, c: torch.Tensor,
               width: int, out_width: int | None = None):
    """Take the final ``width`` tokens and left-pad them to ``out_width``."""
    out = int(width if out_width is None else out_width)
    n = v.shape[1]
    keep = min(int(width), n)
    lead = out - keep
    v2, m2, c2 = v[:, n - keep:], m[:, n - keep:], c[:, n - keep:]
    if lead <= 0:
        return v2, m2, c2
    z = v.new_zeros((v.shape[0], lead))
    mz = m.new_zeros((m.shape[0], lead))
    return (torch.cat([z, v2], dim=1),
            torch.cat([mz, m2], dim=1),
            torch.cat([z, c2], dim=1))


def build_levels(values: torch.Tensor, mask: torch.Tensor, ratios,
                 width, n_levels: int):
    """Build and flatten a coarse-to-fine pyramid.

    ``values`` and ``mask`` have shape ``[B, T]``. ``width`` is either a scalar
    or one width per level. The returned tensors have shape ``[B, L, W_grid]``.
    """
    widths = resolve_level_widths(width, n_levels)
    out_w = max(widths)
    levels = [(values, mask, mask.to(values.dtype))]
    for ratio in ratios:
        pv, pm, _ = levels[-1]
        ratio = int(ratio)
        usable = (pv.shape[1] // ratio) * ratio
        if usable < ratio:
            break
        v2 = pv[:, :usable].reshape(pv.shape[0], usable // ratio, ratio)
        m2 = pm[:, :usable].reshape(pm.shape[0], usable // ratio, ratio)
        count = m2.sum(dim=-1)
        # Mask invalid values before summation. NaN multiplied by zero remains
        # NaN and would otherwise contaminate the pooled level.
        aggregate = (torch.where(m2, v2, torch.zeros_like(v2)).sum(dim=-1)
                     / count.clamp(min=1))
        aggregate = torch.where(count > 0, aggregate, torch.zeros_like(aggregate))
        levels.append((aggregate, count > 0, count.to(values.dtype) / float(ratio)))
    while len(levels) < n_levels:
        levels.insert(0, levels[0])

    values_out, masks_out, coverage_out = [], [], []
    for j, level in enumerate(reversed(levels)):
        v, m, c = _take_tail(level[0], level[1], level[2], widths[j], out_w)
        values_out.append(v)
        masks_out.append(m)
        coverage_out.append(c)
    return (torch.stack(values_out, dim=1),
            torch.stack(masks_out, dim=1),
            torch.stack(coverage_out, dim=1))


def unpad_levels(x: torch.Tensor, widths) -> torch.Tensor:
    """Remove left padding and concatenate each level along the token axis.

    Each level is left-padded to the widest grid width, so valid tokens occupy
    the end of each row. Concatenating the unpadded rows preserves the coarse-to-
    fine ordering and places the most recent samples at the end of the sequence.
    For uniform widths this is equivalent to ``x.reshape(B, L * W, ...)``.
    """
    n_levels, grid_width = x.shape[1], x.shape[2]
    parts = []
    for j in range(n_levels):
        width = int(widths[j])
        if width <= 0 or width > grid_width:
            raise ValueError(
                f"level_widths[{j}]={width} is incompatible with grid width {grid_width}")
        parts.append(x[:, j, grid_width - width:])
    return torch.cat(parts, dim=1)


def window_minmax(values: torch.Tensor, eps: float = 1e-5):
    """Return detached per-window min/max statistics using WindowMinMax semantics.

    Non-finite values are replaced with zero before the reduction. The range is
    clamped to ``eps`` and both statistics are detached. Masked positions still
    contribute to the statistics, matching the reference implementation.
    """
    v = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    v_min = v.min(dim=1, keepdim=True).values.detach()
    v_max = v.max(dim=1, keepdim=True).values.detach()
    v_range = (v_max - v_min).clamp(min=eps).detach()
    return v_min, v_range
