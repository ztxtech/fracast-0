"""Future-state decoder path: seasonal fill followed by causal convolution.

A static pooled summary gives every horizon position the same input and can
only reproduce learned prototype shapes. This module appends the seasonal
fill to the tail of the encoded context and applies one shared causal block
across dilations so each horizon position receives an evolving state.

The projection is zero-initialized, so enabling this path starts from the
static-summary baseline. Causal padding prevents target leakage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fracast._core.self_similar_block import SelfSimilarBlock
from fracast._core.periodic.official_encoding import N_RECENCY_CHANNELS


class FutureConvStates(nn.Module):
    """Map encoded context, seasonal fill, and future features to horizon states."""

    def __init__(self, d: int, n_layers: int, seed: int, kernel: int = 3,
                 ffn_mult: float = 1.5, dilation_base: int = 2):
        super().__init__()
        self.d = int(d)
        self.n_layers = max(1, int(n_layers))
        self.seed = int(seed)
        if self.seed < 1:
            raise ValueError(f"future_conv_seed must be >= 1, got {self.seed}")
        self.dilation_base = int(dilation_base)
        self.in_proj = nn.Linear(1 + N_RECENCY_CHANNELS, self.d)
        # One shared block is reused across all future dilations.
        self.block = SelfSimilarBlock(self.d, int(kernel), float(ffn_mult),
                                      causal=True, separable=True)
        self.out_proj = nn.Linear(self.d, self.d)
        # Zero initialization preserves the baseline exactly at startup.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, fill: torch.Tensor,
                fut_pe: torch.Tensor) -> torch.Tensor:
        """Return horizon states from encoded context, seasonal fill, and features."""
        if h.dim() != 3:
            raise ValueError(f"FutureConvStates expects h=[B,L,D], got {tuple(h.shape)}")
        H = fill.shape[1]
        if fut_pe.shape[:2] != fill.shape:
            raise ValueError(f"fut_pe shape {tuple(fut_pe.shape)} does not match fill shape {tuple(fill.shape)}")
        ft = self.in_proj(torch.cat([fill.unsqueeze(-1), fut_pe], dim=-1).to(h.dtype))
        s = min(self.seed, h.shape[1])
        z = torch.cat([h[:, -s:], ft], dim=1)              # [B, s+H, D]
        for i in range(self.n_layers):
            z = self.block(z, self.dilation_base ** i)
        return self.out_proj(z[:, -H:])
