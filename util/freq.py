"""频率与时间工具（唯一实现：全项目共用，禁止各自再写一份）。

历史问题：freq_to_seconds 曾在 4 处重复实现（model/pyramid.py、
dataport/shard_dataset.py、model/fractal/predictor.py、module/freq.py），
口径漂移风险高，现统一到此文件。

解析口径（2026-09-15 修）：pandas 的 "MS / QS / AS / YS / W-SUN" 等别名
都以字母结尾，不能只用 `endswith("S")` 判秒 ✗ —— 那会把 MS（月初）当成秒级，
把 Q-DEC / A-DEC / W-SUN 落到兜底值。这里先拆出「数字 + 单位」，再按单位表映射；
未知串沿用训练侧历史兜底 3600s ✓。
"""
from __future__ import annotations

import re

import numpy as np

_FREQ_RE = re.compile(r"^(?P<n>\d*)(?P<unit>[A-Za-z]+)")
_UNIT_SECONDS = {
    "S": 1,
    "T": 60, "MIN": 60,
    "H": 3600,
    "D": 86400, "B": 86400,
    "W": 604800,
    "M": 2592000, "MS": 2592000, "ME": 2592000,
    "Q": 7889400, "QS": 7889400, "QE": 7889400,
    "A": 31557600, "AS": 31557600, "YS": 31557600,
    "Y": 31557600, "YE": 31557600,
}


def _unit_seconds(unit: str) -> int:
    """单位串 → 秒；逐字符去掉 B/C/S 等 pandas 变体前缀（BMS → MS ✓）。"""
    u = unit
    while len(u) > 1 and u not in _UNIT_SECONDS:
        u = u[1:]
    return _UNIT_SECONDS.get(u, 3600)


def freq_to_seconds(freq: str) -> int:
    """GIFT-Eval / 语料频率串 → 每步秒数（构造合成 unix 时间戳用）。

    例：S=1、10S=10、15T=900、6H=21600、D=86400、W-SUN=604800、
    M=MS=2592000、Q-DEC=7889400、A-DEC=31557600；未知 "?" 沿用 3600s ✓。
    """
    f = str(freq).strip().upper()
    m = _FREQ_RE.match(f)
    if not m:
        return 3600
    n = int(m.group("n") or "1")
    return n * _unit_seconds(m.group("unit"))


def get_seasonality(freq: str) -> int:
    """Return a compact seasonal period for common time-series frequencies.

    This mirrors the values needed by the pretraining data path without pulling
    in pandas or GluonTS. Frequencies not listed here fall back to one step.
    """
    f = str(freq).strip().upper()
    if f in {"S", "4S", "10S"}:
        return 60
    if f == "T":
        return 24 * 60
    if f.endswith("T"):
        return max(1, int(24 * 60 / int(f[:-1] or "1")))
    if f == "H":
        return 24
    if f.endswith("H"):
        return max(1, int(24 / int(f[:-1] or "1")))
    if f == "D":
        return 7
    if f.endswith("D"):
        return 7
    if f == "W" or "W-" in f:
        return 1
    if f == "M" or f == "MS" or "M-" in f:
        return 12
    if "Q" in f:
        return 4
    if "A" in f or "Y" in f:
        return 1
    return 1


def start_to_unix_seconds(start) -> int:
    """gluonts start（pandas Timestamp / datetime64 / str）→ unix 秒。"""
    try:
        return int(np.datetime64(start, "s").astype("int64"))
    except Exception:
        return 0
