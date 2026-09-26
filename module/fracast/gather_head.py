"""Fracast forecast head: context summary, horizon queries, and causal smoothing.

The head pools a full-resolution context stream, builds horizon queries, applies
residual SwiGLU decoding, and optionally runs a causal dilated convolution along
the forecast horizon. Future-state injection is zero-initialized, so enabling it
starts from the static-summary baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.fracast.self_similar_block import SwiGLU
from module.fracast.quant import fake_quant_act_dynamic
from module.periodic.official_encoding import _norm_fp32, _recency_encoding
from module.fracast.future_conv import FutureConvStates
from module.periodic.seasonal_fill import (detect_periods,
                                           folded_seasonal_fill,
                                           last_period_fill)


class HorizonConvBlock(nn.Module):
    """Causal depthwise dilated convolution over the horizon with an RMSNorm residual."""

    def __init__(self, d: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.d = int(d)
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.int8_dynamic_act = False
        self.dw_weight = nn.Parameter(torch.empty(self.d, 1, self.kernel))
        self.dw_bias = nn.Parameter(torch.zeros(self.d))
        nn.init.kaiming_uniform_(self.dw_weight, a=5 ** 0.5)
        self.norm = nn.RMSNorm(self.d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel - 1) * self.dilation
        xt = F.pad(x.transpose(1, 2), (pad, 0))
        if self.int8_dynamic_act:
            xt = fake_quant_act_dynamic(xt)
        y = F.conv1d(xt, self.dw_weight, self.dw_bias,
                     dilation=self.dilation, groups=self.d).transpose(1, 2)
        if self.int8_dynamic_act:
            y = fake_quant_act_dynamic(y)
        return _norm_fp32(self.norm, x + y)


class GatherQuantileHead(nn.Module):
    """Map a normalized context ``[B, L, D]`` to horizon quantiles ``[B, H, Q]``."""

    def __init__(self, d_model: int, quantiles: list[float], horizon: int,
                 n_queries: int, steps_per_query: int, d_ff: int,
                 decoder_depth: int = 1, fc_layers: int = 2, fc_kernel: int = 3,
                 output_mode: str = "direct", *,
                 future_conv: bool = False, future_conv_layers: int = 6,
                 future_conv_seed: int = 128, phase_bins: int = 16,
                 period_topk: int = 4, period_min: int = 2,
                 period_alpha: float = 0.05, ffn_mult: float = 1.5,
                 kernel: int = 3, dilation_base: int = 2,
                 seasonal_fill_mode: str = "phase_mean"):
        super().__init__()
        d = int(d_model)
        self.d = d
        self.horizon = int(horizon)
        self.n_queries = int(n_queries)
        self.steps_per_query = int(steps_per_query)
        if self.horizon != self.n_queries * self.steps_per_query:
            raise ValueError(
                f"horizon({self.horizon}) must equal "
                f"n_queries({self.n_queries}) * steps_per_query({self.steps_per_query})")
        self.output_mode = str(output_mode)
        self.head_pool = "mean"
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32),
                             persistent=False)
        n_q = len(quantiles)

        # Context summary and horizon queries.
        self.summary_proj = nn.Linear(2 * d + 5, d)
        self.queries = nn.Parameter(torch.zeros(self.horizon, d))
        nn.init.normal_(self.queries, std=d ** -0.5)

        # Residual SwiGLU decoding layers.
        self.decoder_ffns = nn.ModuleList(
            [SwiGLU(d, int(d_ff)) for _ in range(int(decoder_depth))])
        self.decoder_norms = nn.ModuleList(
            [nn.RMSNorm(d) for _ in range(int(decoder_depth))])

        # Causal dilated convolution along the horizon.
        self.fc_blocks = nn.ModuleList(
            [HorizonConvBlock(d, fc_kernel, 2 ** i) for i in range(int(fc_layers))])

        self.out_proj = nn.Linear(d, n_q)

        # Optional future-state path with a zero-initialized residual.
        self.future_conv_on = bool(future_conv)
        self.seasonal_fill_mode = str(seasonal_fill_mode)
        if self.seasonal_fill_mode not in {"phase_mean", "last_period"}:
            raise ValueError(
                f"unknown model.head_seasonal_fill_mode: {self.seasonal_fill_mode!r}; "
                "expected phase_mean or last_period")
        self.phase_bins = int(phase_bins)
        self.period_kw = dict(top_k=int(period_topk), min_period=int(period_min),
                              alpha=float(period_alpha))
        self.future_conv = (FutureConvStates(d, future_conv_layers,
                                             future_conv_seed, kernel,
                                             ffn_mult, dilation_base)
                             if self.future_conv_on else None)

    def forward_horizon(self, h: torch.Tensor, last_obs: torch.Tensor | None = None,
                        anchor: torch.Tensor | None = None,
                        token_weight: torch.Tensor | None = None,
                        ctx: torch.Tensor | None = None,
                        ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return horizon quantiles from encoded context.

        ``ctx`` and ``ctx_mask`` are required only when future-state smoothing is
        enabled.
        """
        del token_weight
        B, L, D = h.shape
        if D != self.d:
            raise ValueError(f"context dimension {D} does not match head.d_model {self.d}")
        summary = torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)          # [B, 2D]
        dev = h.device
        pos = torch.arange(L, L + self.horizon, device=dev).unsqueeze(0).expand(B, -1)
        pe = _recency_encoding(pos, L).to(h.dtype)                      # [B, H, 5]
        summ = summary.unsqueeze(1).expand(B, self.horizon, D * 2)      # [B, H, 2D]
        x = self.summary_proj(torch.cat([pe, summ], dim=-1))
        x = x + self.queries.unsqueeze(0).to(x.dtype)
        for ffn, norm in zip(self.decoder_ffns, self.decoder_norms):
            x = _norm_fp32(norm, x + ffn(x))
        for blk in self.fc_blocks:
            x = blk(x)
        if self.future_conv is not None:
            # Inject future states before the output projection.
            if ctx is None:
                raise ValueError(
                    "model.head_future_conv is enabled but forward_horizon "
                    "did not receive ctx")
            if ctx.shape[0] != B or ctx.shape[1] != L:
                raise ValueError(
                    f"ctx shape {tuple(ctx.shape)} does not match encoder output [B={B}, L={L}]")
            periods = detect_periods(ctx, ctx_mask, **self.period_kw)
            if self.seasonal_fill_mode == "last_period":
                fill = last_period_fill(ctx, pos, periods, self.phase_bins,
                                        ctx_mask)
            else:
                fill = folded_seasonal_fill(ctx, pos, periods, self.phase_bins,
                                            ctx_mask)
            x = x + self.future_conv(h, fill, pe)
        q = self.out_proj(x)                                            # [B, H, Q]
        # Preserve the input dtype; loss computation handles promotion upstream.
        if self.output_mode == "last_obs" and last_obs is not None:
            q = q - q[:, :1, :] + last_obs.unsqueeze(-1)
        if anchor is not None:
            q = q - q[:, :1, :] + anchor.unsqueeze(-1)
        return q
