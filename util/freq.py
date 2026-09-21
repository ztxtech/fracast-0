"""Shared frequency and timestamp helpers.

A single implementation keeps frequency semantics consistent between corpus
construction and model code. Pandas aliases such as ``MS``, ``QS-DEC``, and
``W-SUN`` do not end in a simple seconds suffix, so parsing first extracts the
numeric multiplier and unit and then maps the unit explicitly. Unknown values
retain the historical hourly fallback.
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
    """Strip pandas business-day and start/end variants before mapping a unit."""
    u = unit
    while len(u) > 1 and u not in _UNIT_SECONDS:
        u = u[1:]
    return _UNIT_SECONDS.get(u, 3600)


def freq_to_seconds(freq: str) -> int:
    """Convert a frequency string to seconds per step.

    Examples: ``S`` is 1, ``10S`` is 10, ``15T`` is 900, ``6H`` is 21600,
    ``D`` is 86400, and ``W-SUN`` is 604800. Unknown values use 3600 seconds.
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
    """Convert a GluonTS start timestamp to Unix seconds."""
    try:
        return int(np.datetime64(start, "s").astype("int64"))
    except Exception:
        return 0
