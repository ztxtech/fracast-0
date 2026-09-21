"""分位数损失（跨模型复用）。"""
from __future__ import annotations

import torch


def pinball_loss(pred_q, target, quantiles_t):
    """pred_q [B,Q], target [B]。"""
    diff = target.unsqueeze(-1) - pred_q
    return torch.mean(torch.maximum(quantiles_t * diff,
                                    (quantiles_t - 1) * diff))


def pinball_loss_mask(pred_q, target, quantiles_t, mask=None):
    """带掩码的 pinball loss：只在 mask=True 的位置上平均。

    pred_q [B,Q], target [B], mask [B] (bool)。mask 为空/全 False 时退回全量平均。
    数据端会把缺失位置填 0，若不过滤，模型会被训练去拟合这些假 0。
    """
    diff = target.unsqueeze(-1) - pred_q
    per = torch.maximum(quantiles_t * diff, (quantiles_t - 1) * diff)
    if mask is None:
        return torch.mean(per)
    m = mask.reshape(-1).to(per.dtype).unsqueeze(-1)
    denom = m.sum()
    # 判定留在 GPU 上 ✓：`float(denom)` 会让 host 每步等一次 GPU（同步点打断计算流水，
    # 是单进程训练的主要停顿之一）。denom>0 时 clamp_min(tiny) 与 denom 逐位相同 →
    # 数值不变；denom==0 时由 where 回退全量均值（原语义）✓。
    safe = denom.clamp_min(torch.finfo(per.dtype).tiny)
    return torch.where(denom > 0, (per * m).sum() / safe, torch.mean(per))
