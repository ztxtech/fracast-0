"""FracCast 唯一模型文件：全分辨率流 + 一份共享滤波器跑遍二进尺度阶梯。

本文件只写「块的排布」；块内数值操作在 `module/fraccast/`
（`pipeline/train.py::_build_model` 按 `model.family` 分派）。

## 结构（一句话）

`x[B,L] → in_proj → 共享块 Φ_shared（dilation = 1,2,4,…,2^(N-1) 逐级调用）+ Δ_i 尺度条件 → RMSNorm → h[B,L,D]`

## 与对照臂的关系（唯一自变量 = 跨尺度共享）

  · `model.share_stages: true`（默认）→ **一份** `SelfSimilarBlock`，N 级复用 ← 我们的主张 ✓
  · `model.share_stages: false`      → 每级一份独立块（官方主干的口径）← 对照臂 ✓
  · `model.scale_cond: film|none`    → 是否给每级几十参数的 Δ_i（identity 初始化）✓

## 上下文与「免费加长」

训练 `N=10` 级（RF = 1+(k-1)·Σdilation = 2047）；推理时把 `model.n_stages` 调大
（如 14 → RF 32767）**不需要新参数**——共享块本来就不含尺度专属参数；
`model.W` / `level_widths` 同步调大即可（相位编码维度只由 top-k 决定，与 W 无关 ✓）。
`scale_cond` 吃的是连续尺度坐标 τ=log2(dilation)，所以训练外的级也有定义、且不加参数 ✓。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from module.fraccast.gather_head import GatherQuantileHead
from module.fraccast.self_similar_block import ScaleCondition, SelfSimilarBlock
from module.periodic.encoder import PeriodicPhaseEncoder
from module.periodic.official_encoding import N_RECENCY_CHANNELS, _recency_encoding


class FraccastCore(nn.Module):
    """FracCast 编码器：全分辨率上下文流 → 表示 `[B, L, D]`。"""

    def __init__(self, cfg: dict):
        super().__init__()
        m, p = cfg["model"], cfg["pyramid"]
        self.d = int(m["d_model"])
        self.W = int(m["W"])
        self.n_stages = int(m.get("n_stages", 10))
        self.dilation_base = int(m.get("dilation_base", 2))
        self.share_stages = bool(m.get("share_stages", True))
        self.scale_cond_kind = str(m.get("scale_cond", "film"))
        self.use_recency = bool(m.get("recency_encoding", True))
        self.periodic_phase = bool(m.get("periodic_phase", True))
        # 单流约束：FracCast 的「多尺度」在主干内部（dilation 阶梯），不再走金字塔 token ✓
        ratios = list(p.get("ratios", []) or [])
        self.level_widths = [int(w) for w in (m.get("level_widths") or [self.W])]
        if ratios:
            raise ValueError(
                "FracCast 只吃**单条全分辨率**上下文（pyramid.ratios 必须为空）："
                f"收到 ratios={ratios}。多尺度由主干 dilation 阶梯承担 ✓")
        if len(self.level_widths) != 1:
            raise ValueError(
                f"level_widths 必须只有一个（全分辨率级），收到 {self.level_widths}")
        if self.level_widths[0] != self.W:
            raise ValueError(
                f"level_widths[0]={self.level_widths[0]} 必须等于 model.W={self.W}")

        n_value_ch = 2 + (N_RECENCY_CHANNELS if self.use_recency else 0)
        self.in_proj = nn.Linear(n_value_ch, self.d)
        # 消融点：periodic_phase（零参数周期图相位，官方 TinyCast 同源机制，见模块 docstring）
        self.periodic = PeriodicPhaseEncoder(cfg) if self.periodic_phase else None
        kernel = int(m.get("kernel_size", 3))
        ffn_mult = float(m.get("ffn_mult", 1.5))
        causal = bool(m.get("causal", True))
        separable = bool(m.get("separable_conv", True))
        # 消融点：share_stages（我们的核心主张）
        if self.share_stages:
            self.block = SelfSimilarBlock(self.d, kernel, ffn_mult, causal, separable)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                SelfSimilarBlock(self.d, kernel, ffn_mult, causal, separable,
                                 dilation=self.dilations()[i])
                for i in range(self.n_stages)])
        # 消融点：scale_cond（Δ_i）
        self.scale = (ScaleCondition(self.d, int(m.get("scale_cond_dim", 8)))
                      if self.scale_cond_kind == "film" else None)
        self.norm = nn.RMSNorm(self.d)

    def dilations(self) -> list[int]:
        """二进尺度阶梯：1, 2, 4, …, base^(N-1)（官方 `backbone.py:306-310` 同口径）。"""
        return [self.dilation_base ** i for i in range(self.n_stages)]

    def forward(self, values: torch.Tensor, mask: torch.Tensor, cov: torch.Tensor,
                **kw) -> torch.Tensor:
        """`values/mask/cov`：`[B, 1, W]`（单条全分辨率级）→ `[B, W, D]`。

        额外关键字（`ts_norm` / `t_abs` / `level_ids` / `n_future`）由评测适配器传入，
        本模型不用，明确丢弃 ✓。
        """
        del kw
        if values.dim() != 3 or values.shape[1] != 1:
            raise ValueError(
                f"FracCast 期望单级输入 [B,1,W]，收到 {tuple(values.shape)}")
        if values.shape[2] != self.W:
            raise ValueError(
                f"上下文长度 {values.shape[2]} ≠ model.W={self.W}（改上下文请同步改配置 ✓）")
        v = torch.nan_to_num(values[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        m = mask[:, 0].bool()
        c = cov[:, 0].to(v.dtype)
        v = torch.where(m, v, torch.zeros_like(v))
        B, L = v.shape
        # 逐通道特征：[value, coverage] + （可选）5 维 recency；三者都是 [B, L, ·] ✓
        feats = [v.unsqueeze(-1), c.unsqueeze(-1)]
        if self.use_recency:
            pos = torch.arange(L, device=v.device).unsqueeze(0).expand(B, -1)
            feats.append(_recency_encoding(pos, L).to(v.dtype))
        x = self.in_proj(torch.cat(feats, dim=-1))
        if self.periodic is not None:
            x = x + self.periodic(v, m, self.level_widths)
        for i, dil in enumerate(self.dilations()):
            xi = x if self.scale is None else self.scale(x, dil)
            x = self.block(xi, dil) if self.share_stages else self.blocks[i](xi, dil)
        return self.norm(x)


class Model(nn.Module):
    """TEFN 风格对外模型：一个 configs 对象带进全部参数。"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        cfg = getattr(configs, "cfg", None) or {"model": {}, "pyramid": {}}
        self.core = FraccastCore(cfg)
        self.head = build_head(cfg)

    def forward(self, values, mask, cov, **kw):
        return self.core(values, mask, cov, **kw)


def build_head(cfg: dict) -> nn.Module:
    """装配 FracCast 的预测头。"""
    m = cfg["model"]
    # 守卫：future_conv 头要拿**一条真实时间序列**去跑周期检测与季节折叠；
    # 金字塔模型展平后的 token 序列不是一条序列 → 直接拒绝，不静默算错 ✗
    if bool(m.get("head_future_conv", False)) and list(
            (cfg.get("pyramid") or {}).get("ratios") or []):
        raise ValueError(
            "model.head_future_conv 只支持单级全分辨率上下文（FracCast，pyramid.ratios 空）；"
            "金字塔展平的 token 序列不是一条真实时间序列，周期检测/季节填充都无意义 ✗")
    kind = str(m.get("head_kind", "gather"))
    if kind != "gather":
        raise ValueError(
            f"unsupported model.head_kind: {kind!r}; "
            "this release ships only the gather head")
    args = (int(m["d_model"]), list(m["quantiles"]), int(m["horizon"]),
            int(m["future_queries"]), int(m["future_patch"]))
    return GatherQuantileHead(
        *args, int(m["head_d_ff"]), int(m.get("head_layers", 1)),
        int(m.get("head_fc_layers", 2)), int(m.get("head_fc_kernel", 3)),
        str(m.get("head_output_mode", "direct")),
        future_conv=bool(m.get("head_future_conv", False)),
        future_conv_layers=int(m.get("head_future_conv_layers", 6)),
        future_conv_seed=int(m.get("head_future_conv_seed", 128)),
        phase_bins=int(m.get("head_phase_bins", 16)),
        period_topk=int(m.get("periodic_topk", 4)),
        period_min=int(m.get("periodic_min_period", 2)),
        period_alpha=float(m.get("periodic_alpha", 0.05)),
        ffn_mult=float(m.get("ffn_mult", 1.5)),
        kernel=int(m.get("kernel_size", 3)),
        dilation_base=int(m.get("dilation_base", 2)),
        seasonal_fill_mode=str(m.get("head_seasonal_fill_mode", "phase_mean")))


def as_configs(cfg: dict):
    from types import SimpleNamespace

    ns = SimpleNamespace()
    for key, value in cfg.items():
        setattr(ns, key, value)
    ns.cfg = cfg
    return ns


def build_from_cfg(cfg: dict):
    """pipeline 统一装配入口 → (core, head)。"""
    model = Model(as_configs(cfg))
    return model.core, model.head
