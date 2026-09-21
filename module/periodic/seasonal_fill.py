"""季节折叠填充 + 周期检测（**零参数**）—— 官方解码端 `_seasonal_naive` 的移植。

## 为什么需要它（2026-09-17 定位到的短板根因）

我们的解码头 `GatherQuantileHead` 把整条上下文压成**一个向量**再复制到每个未来位置
（`summary.unsqueeze(1).expand(B, H, ...)`）→ 每个未来步看到的是同一个向量，形状只能来自
"全局学到的原型"。实测季节振幅还原率：restaurant 0.02 / hierarchical_sales 0.03 /
m4_daily 0.19 / solar 1.00 —— 只有"全样本共享同一日循环"的 solar 能出形状 ✓。

官方解码端靠两件事把形状带上：
  ① 把上下文按显著周期**折叠成逐相位均值**，再按未来步的相位取回（**本文件**，零参数）；
  ② 把这条填充喂进因果膨胀卷积，让波形沿地平线自己往前走（`future_conv.py`）。

## 官方出处（逐行对照）

  · https://github.com/raws-labs/tinycast `tinycast/backbone.py` `_seasonal_naive`
  · 同文件 `_detect_periods`（周期图，带 `@torch.compiler.disable`）
我们复用的是本仓库已有的**逐字官方副本** `official_periodogram.significant_periods` ✓。

## 与官方的两处必要偏离（登记）

1. **支持 mask**：官方上下文无缺失；我们 30/97 个配置含 NaN → 相位均值只用**有效点**求，
   无效点不参与计数（官方的 `cnt` 是全部点）。
2. **空相位箱回退**：官方回退到全体均值；我们回退到**该行有效点均值** —— 有缺失值时这是
   同一个量的正确版本（无缺失时两者逐位相同 ✓，见 `script/tests/test_head_future_conv.py`）。

其余（`phase_bins=16`、按主周期 `p0 = periods[:, 0]` 折叠、`round`→`long` 的箱号、
`clamp(max=nb-1)`）一律照官方 ✓。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from module.periodic.encoder import _fill_nan_row
from module.periodic.official_periodogram import significant_periods


@torch.compiler.disable()
def detect_periods(x: torch.Tensor, mask: torch.Tensor | None = None, *,
                   top_k: int = 4, min_period: int = 2,
                   alpha: float = 0.05) -> torch.Tensor:
    """逐条序列跑官方周期图，返回显著周期的 top-k，形状 `[B, K]`（long）。

    `@torch.compiler.disable`：周期图里的 `rfft` 是复数运算，Inductor 编译不了；
    留在编译图里会让整个前向每步退化回 eager + 图断开（官方同处理 ✓）。
    无显著周期时官方按 0 填充槽位，这里原样保留、由下游 `clamp(min=1)` 兜底 ✓。
    """
    B, L = x.shape
    v = _fill_nan_row(x.float(), mask)
    periods, _scores, _n = significant_periods(
        v, min_period=int(min_period), max_period=max(2, L // 2),
        top_k=int(top_k), significance_alpha=float(alpha))
    return periods


def folded_seasonal_fill(x: torch.Tensor, fut_pos: torch.Tensor,
                         periods: torch.Tensor, phase_bins: int = 16,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
    """把上下文按**主周期**折叠成逐相位均值，再按未来步相位取回。

    `x`       `[B, L]`  归一化后的上下文（与模型输入同一归一化空间 ✓）
    `fut_pos` `[B, H]`  未来步的绝对位置（本模型 = `arange(L, L+H)`）
    `periods` `[B, K]`  显著周期（第 0 列 = 主周期，来自 `detect_periods` ✓）
    返回      `[B, H]`  季节 naive 填充值
    """
    if x.dim() != 2:
        raise ValueError(f"folded_seasonal_fill 期望 [B, L]，收到 {tuple(x.shape)}")
    B, L = x.shape
    nb = int(phase_bins)
    if nb < 1:
        raise ValueError(f"phase_bins 必须 ≥1，收到 {nb}")
    dt = x.dtype
    w = torch.ones_like(x) if mask is None else mask.to(dt)
    xv = torch.where(w > 0, torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                     torch.zeros_like(x))

    t = torch.arange(L, device=x.device).view(1, L).float()
    p0 = periods[:, :1].clamp(min=1).float()                       # (B,1) 主周期
    pbin = torch.clamp(((t % p0) / p0 * nb).long(), max=nb - 1)    # (B,L)
    oh = F.one_hot(pbin, nb).to(dt) * w.unsqueeze(-1)              # (B,L,nb) 只数有效点
    cnt = oh.sum(dim=1)                                            # (B,nb)
    base = torch.bmm(oh.transpose(1, 2), xv.unsqueeze(-1)).squeeze(-1)   # (B,nb)
    gmean = (xv.sum(dim=1, keepdim=True)
             / w.sum(dim=1, keepdim=True).clamp(min=1.0))          # (B,1) 有效点均值
    base = torch.where(cnt > 0, base / cnt.clamp(min=1.0), gmean.expand(B, nb))
    fb = torch.clamp((fut_pos.float() % p0) / p0 * nb, max=nb - 1).long()  # (B,H)
    return torch.gather(base, 1, fb)                               # (B,H)


def last_period_fill(x: torch.Tensor, fut_pos: torch.Tensor,
                     periods: torch.Tensor, phase_bins: int = 16,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """复制最近一个完整周期里同相位的点；无效源点回退到全行有效均值。

    与 `folded_seasonal_fill` 相比，这里不做相位平均：每个未来步直接取
    `fut_pos - ceil((fut_pos - L + 1) / p0) * p0`，保证源点位于当前上下文尾部、
    且与未来点同相位。这对方形 / 阶跃平台尤其重要：相位均值会把高低平台折成中间态。
    `phase_bins` 保留在签名里是为了让两种填充在头部可互换；本函数不用它。
    """
    if x.dim() != 2:
        raise ValueError(f"last_period_fill 期望 [B, L]，收到 {tuple(x.shape)}")
    B, L = x.shape
    dt = x.dtype
    w = torch.ones_like(x) if mask is None else mask.to(dt)
    xv = torch.where(w > 0, torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
                     torch.zeros_like(x))

    p0 = periods[:, :1].clamp(min=1).long()                       # (B,1)
    # 未来位置至少比上下文末位晚 1 步；向上取整后回到同一相位的最近历史点。
    distance = fut_pos.long() - (L - 1)
    src = fut_pos.long() - ((distance + p0 - 1) // p0) * p0        # (B,H)
    in_range = (src >= 0) & (src < L)
    src_safe = src.clamp(min=0, max=L - 1)
    values = torch.gather(xv, 1, src_safe)
    valid = in_range
    if mask is not None:
        valid = valid & torch.gather(mask.bool(), 1, src_safe)
    gmean = (xv.sum(dim=1, keepdim=True)
             / w.sum(dim=1, keepdim=True).clamp(min=1.0))          # (B,1)
    return torch.where(valid, values, gmean.expand(B, values.shape[1])).to(dt)
