"""周期图相位编码的**集成层**（我们自己的新代码 —— 算法本体在 official_*.py，逐字官方）。

## 移植来源与偏离声明

官方出处：`raws-labs/tinycast`（Apache-2.0，arXiv:2608.15767）
  · 检测器 `tinycast/periodogram.py::significant_periods` —— 本仓库逐字复制为
    `module/periodic/official_periodogram.py`（零修改，正文后缀逐字校验通过）
  · 相位编码 `tinycast/encoding.py::_phase_encoding` —— 逐字复制为
    `module/periodic/official_encoding.py`

**我们新增的（官方没有的）**：把上述两个函数接到 fractal 的**金字塔多分辨率**输入上。
具体有三处集成决策，均为有意偏离，必须单独看：

1. **逐级独立检测**。官方只有一条 stride=1 的上下文，在其上跑一次检测。我们的输入是
   3 个不同步长的视图（展平顺序「粗级在前、最细级在后」），所以**每一级用自己的序列各跑一次**，
   位置用该级自己的步长单位 —— 这样「周期」在该级内部自然成立，无需跨级换算。
2. **不移植 `_recency_encoding`**（官方 5 维有界记忆基）。理由：本次 OFAT 只测「周期算出来」
   这一个机制；recency 是另一个机制（且我们已有 time_rope + 日历特征承担时间位置），
   混在一起就分不清是谁的功劳。
3. **不移植官方的 `period_trust` 门控与 `phase_bins` 折叠季节剖面**（`backbone.py`
   L231-247 / L341-363）。理由同上：先测核心机制。这两项是明确的后续 OFAT 候选。

集成决策之外的算法本体一律照官方 —— 阈值、Bonferroni 口径、`round` 而非 `truncate`
（官方 L94-99 特别说明：截断会把周期 7 翻成 6，14% 周期误差在 48 步视界上变成整周期相位漂移）、
local-max 过滤、`top_k` 语义，全部来自官方文件。

## 数据侧的必要适配（我们的输入含 NaN）

官方上下文无 NaN。我们的语料 30/97 配置含 NaN，检测器里的 `.mean()` 遇 NaN 会整体污染，
故先用**该级自身的有效点均值**填充（全 NaN 则填 0），再做官方算法（官方自己也会 mean-center）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from module.periodic.official_encoding import _phase_encoding
from module.periodic.official_periodogram import significant_periods


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _fill_nan_row(v: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
    """用该行有效点均值填充 NaN/Inf（全无效则填 0），供周期图使用。"""
    v = torch.nan_to_num(v, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
    finite = torch.isfinite(v)
    if m is not None:
        finite = finite & m
    n = finite.sum(dim=1, keepdim=True).clamp(min=1)
    # 无效点的和取 0，再除以有效点数 → 有效均值
    s = torch.where(finite, v, torch.zeros_like(v)).sum(dim=1, keepdim=True)
    mean = s / n
    return torch.where(finite, v, mean)


class PeriodicPhaseEncoder(nn.Module):
    """从上下文**算出**显著周期，把它们变成相位编码（零参数检测 + 一个投影）。

    输出加到编码器输入上；未通过显著性检验的周期槽位按官方语义输出全 0。
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m, p = cfg["model"], cfg["pyramid"]
        self.d = int(m["d_model"])
        self.W = int(m["W"])
        self.top_k = int(m.get("periodic_topk", 16))
        self.n_harm = int(m.get("periodic_harmonics", 1))
        self.min_period = int(m.get("periodic_min_period", 2))
        self.alpha = float(m.get("periodic_alpha", 0.05))

        # 金字塔展平顺序「粗级在前、最细级在后」：行 r 对应 level li = L-1-r，
        # 步长 = prod(ratios[:li])（与 dataport/shard_dataset.py L302-311 一致）。
        ratios = list(p["ratios"])
        self.L = len(ratios) + 1
        strides = []
        for r in range(self.L):
            li = self.L - 1 - r
            s = 1
            for x in ratios[:li]:
                s *= int(x)
            strides.append(s)
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.long),
                             persistent=False)

        # 官方 topk 维度 = min(top_k, n_bins)，n_bins = n_fft//2（跳过 DC 后剩 n_fft/2）。
        n_bins = _next_pow2(max(2, self.W)) // 2
        self.K = min(self.top_k, n_bins)
        self.proj = nn.Linear(2 * self.K * self.n_harm, self.d)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None,
                widths) -> torch.Tensor:
        """values/mask: [B, N] **扁平金字塔**（粗级在前）→ 相位特征 [B, N, d]。

        `widths`（粗级在前）= 每级 token 数：每一级用**自己的那段序列**跑一次检测，
        位置用该级自己的步长单位（集成决策 #1 ✓）。去 padding 后的扁平输入保证
        检测不会被补零污染，也不会为 padding 白算 ✓。
        """
        B, N = values.shape
        widths = [int(w) for w in widths]
        if len(widths) != self.L:
            raise ValueError(f"PeriodicPhaseEncoder 期望 L={self.L}（金字塔级数），收到 {len(widths)}")
        if sum(widths) != N:
            raise ValueError(f"宽度之和 {sum(widths)} 与 token 数 {N} 不符")
        if max(widths) != self.W:
            raise ValueError(
                f"最细级宽度 {max(widths)} 必须等于 model.W={self.W}（检测器 K 由它定）")
        dev = values.device
        feats = values.new_zeros((B, N, 2 * self.K * self.n_harm))
        off = 0
        for r, wr in enumerate(widths):
            v = values[:, off:off + wr].float()
            m = mask[:, off:off + wr].bool() if mask is not None else None
            v = _fill_nan_row(v, m)
            pos = torch.arange(wr, device=dev, dtype=torch.long)
            pos = pos.unsqueeze(0).expand(B, wr)
            # 检测器无可学习参数（周期是 round→long 的整数），no_grad 语义等价且省图。
            with torch.no_grad():
                periods, _scores, _n_valid = significant_periods(
                    v, min_period=self.min_period, max_period=max(2, wr // 2),
                    top_k=self.top_k, significance_alpha=self.alpha)
            if periods.shape[1] != self.K:
                raise ValueError(
                    f"检测器返回 K={periods.shape[1]}，与初始化时算出的 {self.K} 不符")
            feats[:, off:off + wr] = _phase_encoding(pos, periods,
                                                     n_harmonics=self.n_harm)
            off += wr
        return self.proj(feats)                        # [B, N, d]
