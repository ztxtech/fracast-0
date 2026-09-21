"""Quantile loss functions."""
from __future__ import annotations

import torch


def pinball_loss(pred_q, target, quantiles_t):
    """Compute the pinball loss for predictions shaped ``[B, Q]`` and targets ``[B]``."""
    diff = target.unsqueeze(-1) - pred_q
    return torch.mean(torch.maximum(quantiles_t * diff,
                                    (quantiles_t - 1) * diff))


def pinball_loss_mask(pred_q, target, quantiles_t, mask=None):
    """Compute the pinball loss only at positions where ``mask`` is true.

    ``pred_q`` has shape ``[B, Q]``, ``target`` has shape ``[B]``, and ``mask``
    has shape ``[B]``. Missing target values are filled with zero by the data
    pipeline, so they must be excluded from the loss. If ``mask`` is omitted or
    contains no true values, the function falls back to the unmasked mean.
    """
    diff = target.unsqueeze(-1) - pred_q
    per = torch.maximum(quantiles_t * diff, (quantiles_t - 1) * diff)
    if mask is None:
        return torch.mean(per)
    m = mask.reshape(-1).to(per.dtype).unsqueeze(-1)
    denom = m.sum()
    # Keep the decision on the accelerator. Converting denom to a Python float
    # would synchronize the device on every training step. When denom is
    # positive, clamping it to the dtype minimum leaves the value unchanged;
    # when it is zero, the where branch preserves the unmasked fallback.
    safe = denom.clamp_min(torch.finfo(per.dtype).tiny)
    return torch.where(denom > 0, (per * m).sum() / safe, torch.mean(per))
