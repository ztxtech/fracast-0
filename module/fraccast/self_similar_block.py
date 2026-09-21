"""FracCast 的核心块：一份滤波器 Φ_shared + 每级尺度条件 Δ_i。

## 主张

时间序列在时间尺度上近似自相似 → 同一份滤波算子应当被复用到整条二进尺度阶梯上。
本文件把这个主张落成**一个可实例化的块**：`SelfSimilarBlock` 只持有**一份**参数，
`forward` 每次接受一个 `dilation`（= 该级的时间尺度），于是同一份权重服务所有尺度 ✓。

## 官方语义出处（只读快照，逐行对照）

  · https://github.com/raws-labs/tinycast `tinycast/backbone.py` `_SwiGLU`
  · 同文件 `_DilatedConvBlock` —— 因果左填充、depthwise 可分离、两条 RMSNorm 残差
  · 同文件前向（`x = norm1(x + conv(x))`；`x = norm2(x + ffn(x))`）
  · https://github.com/raws-labs/tinycast `tinycast/encoding.py` `_norm_fp32`

## 我们的偏离（有意，且必须能被消融关掉）

1. **dilation 由 forward 传入**，不再是构造期常量 —— 这是「一份权重服务所有尺度」的实现前提。
    官方每个块有自己的 dilation 常量，等价于我们的 `share_stages: false` 对照臂 ✓。
2. **Δ_i 尺度条件**（`ScaleCondition`）：按时间尺度给出的 FiLM，identity 初始化 →
   训练起点**严格等于纯共享**；Δ_i 的幅度本身是可测量量（论文 C2 用）✓。
3. SwiGLU 的 `d_hidden = int(d * ffn_mult)` 与官方一致（官方默认 `ffn_mult=1.5`，`model.py:33-42`）。

与项目内既有块 `module/trunks/shared_conv.py::ConvFFNBlock` 的差别（所以没有复用）：
GELU → SwiGLU、单条 norm → 两条 norm、dilation 参数化方式不同。两者都保留，各自服务自己的模型 ✓。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.periodic.official_encoding import _norm_fp32
from module.fraccast.quant import fake_quant_act_dynamic


class SwiGLU(nn.Module):
    """官方 `_SwiGLU`（`backbone.py:38-49`）同语义：`down(silu(gate) * val)`。"""

    def __init__(self, d: int, d_hidden: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * int(d_hidden))
        self.down = nn.Linear(int(d_hidden), d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class SelfSimilarBlock(nn.Module):
    """Φ_shared：膨胀因果 depthwise 可分离卷积 + SwiGLU，两条 RMSNorm 残差。

    输入输出都是 `[B, L, D]`（层序在前），与项目其它块一致 ✓。
    `dilation` 是**前向参数**：同一个实例可以用 1, 2, 4, …, 2^(N-1) 依次调用，
    这就是「同一算子在所有尺度上复用」的落地方式 ✓。
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
            # depthwise（每通道一组）+ 1×1 点卷积（= 逐 token 线性混合）✓ 官方 `:82-91`
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
            raise ValueError(f"dilation 必须 ≥1，收到 {dil}")
        # 非因果（居中填充）时左右拆分的口径同官方 `:109-113`（右填充占余数）✓
        # 因果：左侧补 (k-1)*dilation 个零（官方 `backbone.py:109-112`）✓
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
    """Δ：按**时间尺度**给出的 FiLM 条件，identity 初始化。

    `x ← (1 + γ) ⊙ x + β`，`(γ, β) = MLP(定频 sin/cos(τ))`，`τ = log2(dilation)`。

    为什么这样设计：
      · 自相似只是**近似**成立，级间「不像」的部分要显式吸收 → 用最省参数的方式给出 Δ ✓；
      · 尺度坐标是**连续的**（τ 而非级号）→ 同一个实例能处理训练时没见过的级，
        且参数量与级数**严格无关**（外推 C3 的结构前提）✓；
      · `to_gb` 零初始化 → 起点严格 identity，于是「共享是否有用」不会被 Δ 掩盖 ✓。
    """

    def __init__(self, d: int, dim: int = 8):
        super().__init__()
        self.d = int(d)
        self.n_freq = max(1, int(dim) // 2)
        self.to_gb = nn.Linear(2 * self.n_freq, 2 * self.d)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def scale_feat(self, dilation: int, *, device=None, dtype=None) -> torch.Tensor:
        """τ = log2(dilation) 的定频 sin/cos 特征（零参数，超范围尺度同样有定义 ✓）。"""
        tau = math.log2(float(max(1, int(dilation))))
        k = torch.arange(self.n_freq, device=device, dtype=torch.float32)
        ang = (tau / torch.pow(2.0, k)) * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)]).to(dtype)

    def forward(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        f = self.scale_feat(dilation, device=x.device, dtype=self.to_gb.weight.dtype)
        gamma, beta = self.to_gb(f).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def gamma_beta(self, dilations: list[int]) -> torch.Tensor:
        """返回 [len(dilations), 2d] 的 (γ, β)，供诊断「Δ 有没有学到东西」✓。"""
        w = self.to_gb.weight
        rows = [self.to_gb(self.scale_feat(int(d), device=w.device, dtype=w.dtype))
                for d in dilations]
        return torch.stack(rows)
