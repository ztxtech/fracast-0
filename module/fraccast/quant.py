"""Post-training INT8 fake quantization for FracCast.

Weights use symmetric per-output-channel quantization. ``w8`` quantizes weights
only; ``w8a8`` also enables per-tensor dynamic activation quantization. RMSNorm,
periodogram features, normalization statistics, and biases remain floating point.
FracCast implements depthwise convolution with ``F.conv1d`` and a ``dw_weight``
parameter, so those weights are handled explicitly alongside Linear and Conv1d.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_QMIN, _QMAX = -128, 127


@torch.no_grad()
def fake_quant_weight_per_outchannel(weight: torch.Tensor) -> torch.Tensor:
    """Apply symmetric INT8 fake quantization per output channel."""
    reduce_dims = tuple(d for d in range(weight.dim()) if d != 0)
    amax = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_(min=1e-12)
    scale = amax / _QMAX
    return (torch.round(weight / scale).clamp_(_QMIN, _QMAX)
            * scale).to(weight.dtype)


def fake_quant_act_dynamic(x: torch.Tensor) -> torch.Tensor:
    """Apply per-tensor symmetric INT8 dynamic activation quantization."""
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
    """Apply FracCast INT8 fake quantization in place."""
    mode = mode.strip().lower()
    if mode not in {"w8", "w8a8"}:
        raise ValueError(f"unknown INT8 mode {mode!r}; expected w8 or w8a8")

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
    print(
        f"[quant] INT8 {mode}: {n_linear} Linear/Conv1d weights, "
        f"{n_depthwise} depthwise weights{suffix}",
        flush=True,
    )
    return model
