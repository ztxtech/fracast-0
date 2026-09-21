"""CPU tests for frequency parsing and band classification.

Pandas aliases such as MS, Q-DEC, A-DEC, and W-SUN end with letters. Treating
the suffix S as seconds misclassifies these values and changes pretraining
timestamps and band balancing. These tests pin the shared mappings.

Run with: env -u PYTHONPATH .venv/bin/python script/tests/test_freq.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataport.shard_dataset import freq_to_seconds as shard_freq_to_seconds  # noqa: E402
from pipeline.policies.band_mix import band_of  # noqa: E402
from util.freq import freq_to_seconds  # noqa: E402

fails: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def main() -> int:
    print("1. freq_to_seconds: seconds through yearly aliases")
    cases = {
        "S": 1, "4S": 4, "10S": 10,
        "T": 60, "5T": 300, "15T": 900, "10min": 600, "MIN": 60,
        "H": 3600, "6H": 21600,
        "D": 86400,
        "W": 604800, "W-SUN": 604800,
        "M": 2592000, "MS": 2592000, "ME": 2592000, "BMS": 2592000,
        "Q": 7889400, "Q-DEC": 7889400, "QS": 7889400,
        "A": 31557600, "A-DEC": 31557600, "AS": 31557600, "YS": 31557600,
        "?": 3600,
    }
    for freq, want in cases.items():
        check(f"freq_to_seconds({freq!r})", freq_to_seconds(freq), want)
    check("training-side helper reuses the shared implementation",
          shard_freq_to_seconds is freq_to_seconds, True)

    print("2. band_of: month, quarter, and year aliases are not seconds")
    band_cases = {
        "10S": "second", "4S": "second", "S": "second",
        "5T": "subhour", "15T": "subhour", "MIN": "subhour",
        "H": "hour", "D": "day",
        "W": "week", "W-SUN": "week",
        "M": "month", "MS": "month",
        "Q": "other", "Q-DEC": "other", "QS": "other",
        "A": "other", "A-DEC": "other", "AS": "other", "YS": "other",
    }
    for freq, want in band_cases.items():
        check(f"band_of({freq!r})", band_of(freq), want)

    if fails:
        print(f"\nFAIL: {len(fails)} checks failed: {fails}")
        return 1
    print("\nPASS: all frequency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
