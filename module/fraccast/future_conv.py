"""解码端的「未来状态」通路：季节 naive 填充 → 因果膨胀卷积 → 逐未来位置的状态。

## 解决什么问题（2026-09-17）

解码头原本把上下文压成一个向量复制到每个未来位置（`summary.expand(B,H,...)`）→
**每个未来步的样本相关输入完全相同**，输出只能是"水平 + 一个全局学到的原型形状"。
实测：日频周波动的季节振幅还原率 0.01–0.03（restaurant / hierarchical_sales /
temperature_rain / loop_seattle），而全样本共享同一日循环的 solar 是 1.00 ✓。

本模块提供缺失的那条通路：把「上下文尾 seed 个编码状态」接上「季节 naive 填充的未来段」
拼成一条序列，**整条再跑一遍因果膨胀卷积**，于是每个未来位置都拿到一个**沿地平线自己演化
出来**的状态，而不是一个被复制的常数 ✓。

## 官方出处（逐行对照）

https://github.com/raws-labs/tinycast `tinycast/backbone.py`
  · `:471-490` `_future_conv_states` —— 拼接 `[h[:, -seed:], in_proj([fill, PE])]` 后过卷积、
    取尾部 H 个状态；
  · `:426-467` —— 模块定义：`fc_in = 1 + n_pe`、`fc_seed`、N 层因果膨胀块、
    `fc_out` **零初始化**（⇒ 起点严格等于"静态摘要"基线，不可能把基线变差 ✓）。
  · 官方注释原话：“The rest of the decoder queries a STATIC pooled summary at every horizon
    position, **which is why error grows with horizon**.” —— 正是我们踩的那个坑。

## 与官方的一处有意偏离（登记，必须在论文里说明）

官方用 **N 个各自独立的块**（`future_conv_layers=6`，层间只共享 FFN，≈39.7K 参数）。
我们用**一份** `SelfSimilarBlock` 依次跑 N 个 dilation —— 这正是 FracCast 的主张
（「同一份滤波算子服务整条尺度阶梯」）在解码端的自然延伸，参数从 ≈39.7K 降到 ≈1/N，
且"共享 vs 不共享"可以靠 `model.share_stages` 同款开关单变量对照 ✓。

因果性：`SelfSimilarBlock(causal=True)` 只左侧补零 → 未来位置只看得到更早的位置，
不会从目标段泄露信息 ✓（`script/tests/test_head_future_conv.py` 有断言）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from module.fraccast.self_similar_block import SelfSimilarBlock
from module.periodic.official_encoding import N_RECENCY_CHANNELS


class FutureConvStates(nn.Module):
    """`(h, fill, fut_pe)` → `[B, H, D]`：逐未来位置的因果卷积状态。"""

    def __init__(self, d: int, n_layers: int, seed: int, kernel: int = 3,
                 ffn_mult: float = 1.5, dilation_base: int = 2):
        super().__init__()
        self.d = int(d)
        self.n_layers = max(1, int(n_layers))
        self.seed = int(seed)
        if self.seed < 1:
            raise ValueError(f"future_conv_seed 必须 ≥1，收到 {self.seed}")
        self.dilation_base = int(dilation_base)
        self.in_proj = nn.Linear(1 + N_RECENCY_CHANNELS, self.d)
        # 消融点：share_stages（与主干同款开关语义）—— 这里恒为「一份块跑 N 个 dilation」✓
        self.block = SelfSimilarBlock(self.d, int(kernel), float(ffn_mult),
                                      causal=True, separable=True)
        self.out_proj = nn.Linear(self.d, self.d)
        # ★ 零初始化 ⇒ 训练起点严格等于基线（加 0 是逐位恒等 ✓），开关关/开在初始化时等价 ✓
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, fill: torch.Tensor,
                fut_pe: torch.Tensor) -> torch.Tensor:
        """`h[B,L,D]` 编码状态；`fill[B,H]` 季节 naive 填充；`fut_pe[B,H,5]` → `[B,H,D]`。"""
        if h.dim() != 3:
            raise ValueError(f"FutureConvStates 期望 h=[B,L,D]，收到 {tuple(h.shape)}")
        H = fill.shape[1]
        if fut_pe.shape[:2] != fill.shape:
            raise ValueError(f"fut_pe {tuple(fut_pe.shape)} 与 fill {tuple(fill.shape)} 不匹配")
        ft = self.in_proj(torch.cat([fill.unsqueeze(-1), fut_pe], dim=-1).to(h.dtype))
        s = min(self.seed, h.shape[1])
        z = torch.cat([h[:, -s:], ft], dim=1)              # [B, s+H, D]
        for i in range(self.n_layers):                    # 一份块，N 个尺度 ✓
            z = self.block(z, self.dilation_base ** i)
        return self.out_proj(z[:, -H:])                   # 只取未来段的状态
