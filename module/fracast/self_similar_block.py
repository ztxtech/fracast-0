"""Core Fracast block with one shared filter and optional scale conditioning.

The block holds a single set of parameters and receives the dilation as a forward
argument. The same filter can therefore operate at every scale in the ladder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.periodic.official_encoding import _norm_fp32
from module.fracast.quant import fake_quant_act_dynamic


class SwiGLU(nn.Module):
    """SwiGLU projection matching the upstream block definition."""

    def __init__(self, d: int, d_hidden: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * int(d_hidden))
        self.down = nn.Linear(int(d_hidden), d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class SelfSimilarBlock(nn.Module):
    """Dilated causal depthwise-separable convolution followed by SwiGLU.

    The dilation is a forward argument, so one instance can process every scale.
    """

    def __init__(self, d: int, kernel: int = 3, ffn_mult: float = 1.5,
                 causal: bool = True, separable: bool = True, dilation: int = 1):
        super().__init__()
        d = int(d)
        self.d = d
        self.kernel = int(kernel)
        self.causal = bool(causal)
        self.separable = bool(separable)
        self.dilation_default = int(dilation)
        self.int8_dynamic_act = False
        if self.separable:
            # Depthwise temporal filtering followed by a pointwise projection.
            self.dw_weight = nn.Parameter(torch.empty(d, 1, self.kernel))
            self.dw_bias = nn.Parameter(torch.zeros(d))
            self.pw = nn.Linear(d, d)
            nn.init.kaiming_uniform_(self.dw_weight, a=5 ** 0.5)
        else:
            self.conv_weight = nn.Parameter(torch.empty(d, d, self.kernel))
            self.conv_bias = nn.Parameter(torch.zeros(d))
            nn.init.kaiming_uniform_(self.conv_weight, a=5 ** 0.5)
        d_hidden = int(d * float(ffn_mult))
        self.ffn = SwiGLU(d, d_hidden)
        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor, dilation: int | None = None) -> torch.Tensor:
        dil = int(self.dilation_default if dilation is None else dilation)
        if dil < 1:
            raise ValueError(f"dilation must be >= 1, got {dil}")
        # Causal mode pads on the left; non-causal mode centers the padding.
        pad_total = (self.kernel - 1) * dil
        left, right = (pad_total, 0) if self.causal else (pad_total // 2, pad_total - pad_total // 2)
        xt = F.pad(x.transpose(1, 2), (left, right))
        if self.int8_dynamic_act:
            xt = fake_quant_act_dynamic(xt)
        if self.separable:
            dw = F.conv1d(xt, self.dw_weight, self.dw_bias, dilation=dil, groups=self.d)
            if self.int8_dynamic_act:
                dw = fake_quant_act_dynamic(dw)
            conv_out = self.pw(dw.transpose(1, 2))
        else:
            conv_out = F.conv1d(xt, self.conv_weight, self.conv_bias,
                                dilation=dil).transpose(1, 2)
            if self.int8_dynamic_act:
                conv_out = fake_quant_act_dynamic(conv_out)
        x = _norm_fp32(self.norm1, x + conv_out)
        x = _norm_fp32(self.norm2, x + self.ffn(x))
        return x


class ScaleCondition(nn.Module):
    """FiLM conditioning defined on a continuous scale coordinate.

    The mapping starts as identity, so scale conditioning cannot mask the effect
    of parameter sharing during the initial training phase.
    """

    def __init__(self, d: int, dim: int = 8):
        super().__init__()
        self.d = int(d)
        self.n_freq = max(1, int(dim) // 2)
        self.to_gb = nn.Linear(2 * self.n_freq, 2 * self.d)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def scale_feat(self, dilation: int, *, device=None, dtype=None) -> torch.Tensor:
        """Return fixed-frequency sine and cosine features for the scale coordinate."""
        tau = math.log2(float(max(1, int(dilation))))
        k = torch.arange(self.n_freq, device=device, dtype=torch.float32)
        ang = (tau / torch.pow(2.0, k)) * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)]).to(dtype)

    def forward(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        f = self.scale_feat(dilation, device=x.device, dtype=self.to_gb.weight.dtype)
        gamma, beta = self.to_gb(f).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def gamma_beta(self, dilations: list[int]) -> torch.Tensor:
        """Return stacked scale-conditioning parameters for diagnostics."""
        w = self.to_gb.weight
        rows = [self.to_gb(self.scale_feat(int(d), device=w.device, dtype=w.dtype))
                for d in dilations]
        return torch.stack(rows)
