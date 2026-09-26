"""Safetensors layout helpers for Fracast inference releases.

The W8 file stores quantized weights and their FP32 scales.  Norms, biases,
query tensors, and residuals stay in FP32 so loading reproduces the published
W8 evaluation exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

_W8_FORMAT = "fracast-w8-symmetric-per-output-channel"
_QWEIGHT_SUFFIX = ".qweight"
_SCALE_SUFFIX = ".scale"


def is_w8_weight(name: str) -> bool:
    """Return whether a tensor is quantized in the W8 release."""
    if name.endswith(("dw_weight", "conv_weight")):
        return True
    if not name.endswith(".weight"):
        return False
    owner = name.split(".")[:-1]
    return not any(part.startswith("norm") for part in owner)


def dequantize_w8_state(
    state: dict[str, torch.Tensor], quant_map: dict[str, dict[str, Any]]
) -> dict[str, torch.Tensor]:
    """Restore a complete FP32 state dict from the W8 release layout."""
    expected = set(state)
    mapped = {item["qweight"] for item in quant_map.values()}
    mapped.update(item["scale"] for item in quant_map.values())
    reserved = expected - mapped
    if mapped | reserved != expected:
        raise ValueError(
            "W8 state does not match its manifest: "
            f"missing={sorted(mapped - expected)} "
            f"unexpected={sorted(expected - (mapped | reserved))}"
        )

    restored: dict[str, torch.Tensor] = {}
    for name, item in quant_map.items():
        qweight = state[str(item["qweight"])]
        scale = state[str(item["scale"])]
        if qweight.dtype != torch.int8 or int(item["axis"]) != 0:
            raise ValueError(
                f"invalid W8 layout for {name}: dtype={qweight.dtype}, "
                f"axis={item.get('axis')}"
            )
        restored[name] = (qweight.to(torch.float32) * scale).contiguous()
    for name in sorted(reserved):
        restored[name] = state[name].contiguous()
    return restored


def load_w8_safetensors(path: str | Path, manifest_path: str | Path) -> dict[str, torch.Tensor]:
    """Load and dequantize the published W8 safetensors."""
    raw = load_file(str(path), device="cpu")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("format") != _W8_FORMAT:
        raise ValueError(f"unknown W8 manifest format: {manifest.get('format')!r}")
    return dequantize_w8_state(raw, manifest["quant_map"])
