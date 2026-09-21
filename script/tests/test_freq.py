"""频率口径单元门（CPU 可跑）：band_of_freq 与 freq_to_seconds 的边界。

为什么单独立这道门（2026-09-15 实测 bug）：
pandas 的 "MS"（月初）、"Q-DEC"（季末）、"A-DEC"（年末）、"W-SUN" 都以字母
结尾；旧解析只用 `endswith("S")` 判秒，导致 MS 被当成秒级、Q-DEC/A-DEC/W-SUN
落到兜底值 —— 训练侧时间戳与频段配平都会错 ✗。这些别名在官方复现线里真实出现
过，所以必须用固定断言锁住，不能再靠“看起来差不多” ✓。

用法：env -u PYTHONPATH .venv/bin/python script/tests/test_freq.py
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
    print(f"  {'✓' if ok else '✗'} {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def main() -> int:
    print("① freq_to_seconds：秒/分/时/日/周/月/季/年 + 起始别名")
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
    check("训练侧 re-export 同一实现（禁止再抄一份）",
          shard_freq_to_seconds is freq_to_seconds, True)

    print("② band_of：MS/QS/AS 不能被吞进 second")
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
        print(f"\n✗ 失败 {len(fails)} 项：{fails}")
        return 1
    print("\n✓ 频率口径全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
