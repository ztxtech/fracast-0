"""FracCast 解码头：上下文摘要 + horizon query + 沿 horizon 的因果卷积。

## 为什么是这个头

主干输出的是**一整条全分辨率上下文流** `h ∈ [B, L, D]`；预测头要把它压成 `[B, H, Q]`。
官方 TinyCast 的解码头由三部分组成（只读快照 `tinycast/backbone.py:389-470`）：
  ① `pool_kind="mean_last"` 的上下文摘要（`pool_dim = 2D`）；
  ② `query_proj` + `decoder_depth` 个残差 SwiGLU（`:392-396`）；
  ③ 沿 horizon 轴的**因果膨胀卷积** `future_conv`（`:427-454`）。
本文件照此复刻这三件，其余（相位剖面 / recency 剖面 / 跨周期卷积）**有意不做** ——
它们是官方解码端另外三条特征路径，本文档按「一次只改一个变量」的规则留作后续单独 OFAT；
相位信息在编码器**输入端**已经注入（官方同样在输入端注入 `_positional_encoding` ✓）。

## 2026-09-17 补：官方那四条「逐步取内容」的通路里最要紧的一条已补上
实测发现本文件原来只复刻了「静态池化摘要」——**每个未来步看到的样本相关输入完全相同**，
于是形状只能来自全局学到的原型：日频周波动的季节振幅还原率只有 0.01–0.03 ✗。
因此按 config 开关补上 `future_conv`（`head_future_conv`，**默认关**）：
季节 naive 填充 → 因果膨胀卷积 → 逐未来位置状态 → **加性**注入 query（`fc_out` 零初始化，
⇒ 初始化时与旧实现逐位恒等，开关关掉也逐位恒等 ✓）。
仍**未**做的：相位剖面 `phase_mix` / recency 量分箱剖面 / 跨周期卷积 —— 留给后续 OFAT。

发布版只保留这一个头：它用全局摘要直接构造 query，不受主干 dilation 范围限制。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.fraccast.self_similar_block import SwiGLU
from module.fraccast.quant import fake_quant_act_dynamic
from module.periodic.official_encoding import _norm_fp32, _recency_encoding
from module.fraccast.future_conv import FutureConvStates
from module.periodic.seasonal_fill import (detect_periods,
                                           folded_seasonal_fill,
                                           last_period_fill)


class HorizonConvBlock(nn.Module):
    """沿 horizon 轴的因果 depthwise 膨胀卷积 + RMSNorm 残差。

    官方 `future_conv`（`backbone.py:427-454`）是「沿 horizon 的因果膨胀卷积 + 注入解码 query」；
    我们只取「沿 horizon 的因果膨胀 depthwise 卷积」这一件，块内不设 FFN（头不该再扛一份 FFN ✗）。
    """

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
    """`[B, L, D]` 上下文 → `[B, H, Q]` 分位数（归一化空间）。

    所有可调量（horizon / query 数 / 解码层数 / horizon 卷积层数 / FFN 宽度）都来自 configs ✓。
    """

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
                f"horizon({self.horizon}) 必须等于 "
                f"n_queries({self.n_queries}) × steps_per_query({self.steps_per_query})")
        self.output_mode = str(output_mode)
        self.head_pool = "mean"          # 供 predictor 读取（本头不消费 token_weight）✓
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32),
                             persistent=False)
        n_q = len(quantiles)

        # ① 摘要 + query：官方 query 输入 = PE + 池化摘要（`backbone.py:381-388`）✓
        self.summary_proj = nn.Linear(2 * d + 5, d)
        self.queries = nn.Parameter(torch.zeros(self.horizon, d))
        nn.init.normal_(self.queries, std=d ** -0.5)

        # ② 解码：decoder_depth 个残差 SwiGLU（官方 `:392-396`）✓
        self.decoder_ffns = nn.ModuleList(
            [SwiGLU(d, int(d_ff)) for _ in range(int(decoder_depth))])
        self.decoder_norms = nn.ModuleList(
            [nn.RMSNorm(d) for _ in range(int(decoder_depth))])

        # ③ 沿 horizon 的因果膨胀卷积，dilation = 2^i（官方 `:451` 同口径）✓
        self.fc_blocks = nn.ModuleList(
            [HorizonConvBlock(d, fc_kernel, 2 ** i) for i in range(int(fc_layers))])

        self.out_proj = nn.Linear(d, n_q)

        # 消融点：future_conv（官方解码端「把波形沿地平线推出去」的通路；**默认关** ✓）
        # 关掉时整个 head 与历史实现逐位恒等（不多算一步 ✓）；打开且未训练时因 fc_out
        # 零初始化，也与关掉时逐位恒等（加 0 是恒等）→ 不可能把基线变差 ✓。
        self.future_conv_on = bool(future_conv)
        self.seasonal_fill_mode = str(seasonal_fill_mode)
        if self.seasonal_fill_mode not in {"phase_mean", "last_period"}:
            raise ValueError(
                f"未知 model.head_seasonal_fill_mode: {self.seasonal_fill_mode!r}；"
                "应为 phase_mean 或 last_period")
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
        """`h`: `[B, L, D]`；返回 `[B, H, Q]`。

        `token_weight` 本头不用（接口对齐）✓。
        `ctx` / `ctx_mask`：**归一化后**的上下文 `[B, L]` 与有效掩码 —— 只有 `future_conv` 打开时
        才需要（它要拿上下文算显著周期与季节 naive 填充 ✓）；关掉时传 None 即可、不参与计算。
        """
        del token_weight
        B, L, D = h.shape
        if D != self.d:
            raise ValueError(f"上下文维度 {D} ≠ head.d_model {self.d}")
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
            # 季节 naive 填充 → 因果膨胀卷积 → 逐未来位置状态 → **加性注入解码状态**（零初始化）
            # 注入必须在 out_proj **之前**：官方 `backbone.py:1032` 也是加在 D 维状态上 ✓
            if ctx is None:
                raise ValueError(
                    "model.head_future_conv 已开，但 forward_horizon 没拿到 ctx"
                    "（归一化后的上下文）→ 训练/推理侧要把它一并传进来 ✗")
            if ctx.shape[0] != B or ctx.shape[1] != L:
                raise ValueError(
                    f"ctx 形状 {tuple(ctx.shape)} 与编码器输出 [B={B}, L={L}] 不匹配 ✗")
            periods = detect_periods(ctx, ctx_mask, **self.period_kw)
            if self.seasonal_fill_mode == "last_period":
                fill = last_period_fill(ctx, pos, periods, self.phase_bins,
                                        ctx_mask)
            else:
                fill = folded_seasonal_fill(ctx, pos, periods, self.phase_bins,
                                            ctx_mask)
            x = x + self.future_conv(h, fill, pe)
        q = self.out_proj(x)                                            # [B, H, Q]
        # 注：不强行转 fp32 —— 保持与输入同 dtype（bf16 下省显存），损失在上游自行提升 ✓
        if self.output_mode == "last_obs" and last_obs is not None:
            q = q - q[:, :1, :] + last_obs.unsqueeze(-1)
        if anchor is not None:
            q = q - q[:, :1, :] + anchor.unsqueeze(-1)
        return q
