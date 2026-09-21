"""金字塔层构建（torch 侧）—— 训练期 AR rollout 专用。

与 DataPort 的 numpy 实现（`dataport/shard_dataset.py: _build_levels` + `_take_tail`）
**同口径**：自底向上按 ratio 做掩码均值池化，每级取末 W 个 token、不足左侧补 0
（值补 0 / mask 补 False / cov 补 0）；级数不足时在最前面复制最细级 ✓。

为什么需要它：官方 TinyCast 的训练是 **AR rollout** —— 每一块都用「它自己那一段上下文」
重新归一化、重建表示（`tinycast/train.py: _rollout_loss`）。DataPort 只在取数时建一次
金字塔，那只对应 rollout 的第 0 块；后面几块必须在 GPU 上现建 ✓。
"""
from __future__ import annotations

import torch


def resolve_level_widths(width, n_levels: int) -> list[int]:
    """解析「每级 token 宽度」（扁平顺序：**粗级在前、最细级在后**）。

    - `model.W`（标量）= 所有级同宽（历史行为，逐位不变 ✓）；
    - `model.level_widths`（列表）= 逐级指定，长度必须等于级数。

    为什么需要它（2026-09-15 设计复核）：均匀 W 下每级只留「末 W 个 token」→
    2048 窗口里只有最后 128 点保原始分辨率，其余只能看 4x/16x 均值 ✗。
    逐级宽度让**近端全分辨率、远端分层压缩**（TinyCast/Reverso 第一尺度都是 stride=1）✓。
    """
    n = int(n_levels)
    if isinstance(width, (list, tuple)):
        widths = [int(w) for w in width]
        if len(widths) != n:
            raise ValueError(
                f"model.level_widths 需要 {n} 个数（= 金字塔级数），收到 {len(widths)}")
        if any(w < 1 for w in widths):
            raise ValueError(f"model.level_widths 必须为正整数，收到 {widths}")
        return widths
    return [int(width)] * n


def level_span(widths, ratios) -> int:
    """金字塔覆盖的原始步数 = max(每级宽度 x 该级倍率)（粗级在前顺序）。

    用途：定 `ctx_len`。均匀宽度下等价于 `W x prod(ratios)`（逐位不变 ✓）。
    """
    span = int(widths[-1])                       # 最细级：倍率 1
    for j in range(len(widths) - 1):             # 行 j 倍率 = prod(ratios[:n-1-j])
        cum = 1
        for r in list(ratios)[:len(widths) - 1 - j]:
            cum *= int(r)
        span = max(span, int(widths[j]) * cum)
    return span


def _take_tail(v: torch.Tensor, m: torch.Tensor, c: torch.Tensor,
               width: int, out_width: int | None = None):
    """取本级末 `width` 个 token，再左侧补零到 `out_width`（与 numpy 版同口径 ✓）。

    `out_width` = 所有级里最宽的网格宽度；最细级通常就是最宽级，所以扁平化之后
    序列尾部仍是原始分辨率的最近点 ✓（补零只出现在前面）。
    """
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
    """建金字塔 → 展平顺序「粗级在前、最细级在后」的 [B, L, W] 三元组。

    `values`/`mask` 形状 [B, T]（原始分辨率的上下文，已归一化 ✓）。
    `width` = 标量（全级同宽）或逐级宽度列表（粗级在前）；网格宽取最大值 ✓。
    """
    widths = resolve_level_widths(width, n_levels)
    out_w = max(widths)
    levels = [(values, mask, mask.to(values.dtype))]
    for r in ratios:
        pv, pm, _ = levels[-1]
        r = int(r)
        t = (pv.shape[1] // r) * r
        if t < r:
            break
        v2 = pv[:, :t].reshape(pv.shape[0], t // r, r)
        m2 = pm[:, :t].reshape(pm.shape[0], t // r, r)
        cnt = m2.sum(dim=-1)
        # 先把无效位清零再求和：NaN × 0 仍是 NaN，会把整块污染 ✗（同 numpy 版注释）
        agg = (torch.where(m2, v2, torch.zeros_like(v2)).sum(dim=-1)
               / cnt.clamp(min=1))
        agg = torch.where(cnt > 0, agg, torch.zeros_like(agg))
        levels.append((agg, cnt > 0, cnt.to(values.dtype) / float(r)))
    while len(levels) < n_levels:
        levels.insert(0, levels[0])

    vs, ms, cs = [], [], []
    for j, lv in enumerate(reversed(levels)):       # 粗级在前 ✓
        v, m, c = _take_tail(lv[0], lv[1], lv[2], widths[j], out_w)
        vs.append(v)
        ms.append(m)
        cs.append(c)
    return torch.stack(vs, dim=1), torch.stack(ms, dim=1), torch.stack(cs, dim=1)


def unpad_levels(x: torch.Tensor, widths) -> torch.Tensor:
    """[B, L, W_grid, ...] → 每级只取**末 width_j** 个后按 token 轴拼接 ✓。

    约定（全仓库统一）：网格是**左侧补零到最宽级**，所以真实 token 恒在每行尾部；
    去掉 padding 之后扁平序列的尾部 = 最细级（原始分辨率）的最近点 ✓。
    `widths` 均匀时本函数 = 原来的 `x.reshape(B, L*W, ...)`（逐位等价 ✓）。

    为什么需要它：逐级宽度不同时网格里必然有 padding（白算）；模型在进分支 / 主干
    之前先把 padding 去掉，只对真实 token 做计算 ✓。
    """
    l, w = x.shape[1], x.shape[2]
    parts = []
    for j in range(l):
        wj = int(widths[j])
        if wj <= 0 or wj > w:
            raise ValueError(f"level_widths[{j}]={wj} 与网格宽 {w} 不兼容")
        parts.append(x[:, j, w - wj:])
    return torch.cat(parts, dim=1)


def window_minmax(values: torch.Tensor, eps: float = 1e-5):
    """逐窗口 min/max 归一化统计量（detach）✓ 官方 `WindowMinMax` 同口径。

    官方出处：`tinycast/normalization.py::WindowMinMax.transform` ——
    `nan_to_num` 之后**在整窗上**取 min/max（**不看掩码** ✓）、统计量 detach、
    range 下限 1e-5、`x_norm = (x - min) / range ∈ [0, 1]`。
    掩码只用于「该位置算不算损失 / 参不参与池化」，不参与归一化统计 ——
    与官方逐字对齐（gap 位置在官方口径里就是被填成 0 的那个值 ✓）。
    """
    v = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    v_min = v.min(dim=1, keepdim=True).values.detach()
    v_max = v.max(dim=1, keepdim=True).values.detach()
    v_range = (v_max - v_min).clamp(min=eps).detach()
    return v_min, v_range
