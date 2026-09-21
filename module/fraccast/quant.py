"""FracCast 训练后 INT8 fake-quant。

口径逐行对齐官方 TinyCast `tinycast/quant.py`
（https://github.com/raws-labs/tinycast）：
权重使用逐输出通道对称 INT8；`w8` 只量权重，`w8a8` 再开逐张量动态激活量化。
RMSNorm、周期图、归一化统计和 bias 保持浮点。

FracCast 的 depthwise 卷积用 `F.conv1d + dw_weight` 实现，不是 `nn.Conv1d`；
本模块额外覆盖这些 `dw_weight`，避免直接复用官方函数时漏量主干/解码卷积。
"""
from __future__ import annotations

import torch
import torch.nn as nn

_QMIN, _QMAX = -128, 127


@torch.no_grad()
def fake_quant_weight_per_outchannel(weight: torch.Tensor) -> torch.Tensor:
    """按输出通道做对称 INT8 fake-quant；Linear 与 Conv1d 的 axis 0 同义。"""
    reduce_dims = tuple(d for d in range(weight.dim()) if d != 0)
    amax = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_(min=1e-12)
    scale = amax / _QMAX
    return (torch.round(weight / scale).clamp_(_QMIN, _QMAX)
            * scale).to(weight.dtype)


def fake_quant_act_dynamic(x: torch.Tensor) -> torch.Tensor:
    """逐张量对称 INT8 动态激活 fake-quant。"""
    if not torch.is_floating_point(x):
        return x
    amax = x.detach().abs().amax().clamp(min=1e-12)
    scale = amax / _QMAX
    return (torch.round(x / scale).clamp_(_QMIN, _QMAX) * scale).to(x.dtype)


def _linear_pre_hook(_module, inputs):
    if not inputs:
        return None
    return (fake_quant_act_dynamic(inputs[0]),) + tuple(inputs[1:])


def _conv_post_hook(_module, _inputs, output):
    return fake_quant_act_dynamic(output)


def quantize_int8_(model: nn.Module, mode: str = "w8") -> nn.Module:
    """就地应用 FracCast INT8 fake-quant；`mode` 取 `w8` / `w8a8`。"""
    mode = mode.strip().lower()
    if mode not in {"w8", "w8a8"}:
        raise ValueError(f"未知 INT8 mode {mode!r}（应为 w8 或 w8a8）")

    n_linear = n_depthwise = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                module.weight.data.copy_(
                    fake_quant_weight_per_outchannel(module.weight.data))
                n_linear += 1
                if mode == "w8a8":
                    module.register_forward_pre_hook(_linear_pre_hook)
                    if isinstance(module, nn.Conv1d):
                        module.register_forward_hook(_conv_post_hook)
            for weight_name in ("dw_weight", "conv_weight"):
                weight = getattr(module, weight_name, None)
                if not isinstance(weight, torch.Tensor):
                    continue
                weight.data.copy_(fake_quant_weight_per_outchannel(weight.data))
                module.int8_dynamic_act = mode == "w8a8"
                n_depthwise += 1

    suffix = " + per-tensor dynamic activation quant" if mode == "w8a8" else ""
    print(f"[quant] INT8 {mode}: Linear/Conv1d 权重 {n_linear} 个、"
          f"FracCast depthwise 权重 {n_depthwise} 个{suffix}", flush=True)
    return model
