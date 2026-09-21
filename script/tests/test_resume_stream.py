"""断点续跑单元门（CPU 可跑全）：跳批必须与「不中断那条流」**逐位相同**。

为什么要有这道门（用户 2026-09-17：程序必须能停、也必须能接着跑）：
续跑最危险的地方不是权重，而是**数据流对齐**。权重/优化器/步数都能靠存盘解决，
但「第 N 步该看到哪个 batch」如果跳错了，训练照样跑、loss 照样降，只是后面
所有样本顺序都串位 —— 曲线看着正常，实验却已经不可复现 ✗。

实现口径：skip 不是「丢掉前 N 个再重抽」，而是**照常推进 RNG，只是不产出 batch**
（sampler 的抽样序列由 `seed + epoch` 决定，逐 batch 消耗固定）✓。
所以本门用**同一份 fake 语料**跑两遍，断言 `skip 流的第 i 个 == 原流的第 N+i 个`。

用法：env -u PYTHONPATH .venv/bin/python script/tests/test_resume_stream.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataport.shard_dataset import ShardBatchSampler        # noqa: E402
from pipeline.policies.band_mix import BandMixSampler       # noqa: E402


class _FakeBandDS:
    """只提供 band_mix sampler 需要的四个字段（语料本体不参与抽样 ✓）。"""

    def __init__(self, n_rows: int = 4096, n_shards: int = 6):
        rng = np.random.default_rng(0)
        self._freqs = ["5T", "H", "D", "W", "M"]       # 5 个频段，各占一段行
        self._freq_ids = np.repeat(np.arange(len(self._freqs)), n_rows // 5)
        self._shard_ids = rng.integers(0, n_shards, size=self._freq_ids.size)
        self._shard_files = [f"/fake/shard{i}.npy" for i in range(n_shards)]


class _FakeShardDS:
    """只提供 ShardBatchSampler 需要的两个字段 ✓。"""

    def __init__(self, n_rows: int = 4096, n_shards: int = 6):
        rng = np.random.default_rng(1)
        self._shard_ids = rng.integers(0, n_shards, size=n_rows)
        self._shard_files = [f"/fake/shard{i}.npy" for i in range(n_shards)]
        self.fast_layout = False       # 关掉预读线程（测试不碰磁盘 ✓）


def _check(name: str, full: list, skipped: list, skip_n: int) -> None:
    assert len(skipped) == len(full) - skip_n, (
        f"{name}: 跳批后长度不对 —— 期望 {len(full) - skip_n}，实得 {len(skipped)}")
    bad = [i for i, (a, b) in enumerate(zip(full[skip_n:], skipped)) if a != b]
    assert not bad, (f"{name}: 第 {bad[:5]} 个 batch 与原流不逐位相同 ✗ "
                     f"（跳批没有对齐 RNG）")
    print(f"  ✓ {name}：跳 {skip_n}/{len(full)} 批后，剩余 {len(skipped)} 批逐位相同")


def test_band_mix_skip() -> None:
    print("[1/3] band_mix sampler 跳批对齐")
    cfg = {"data": {"band_mix": {"shares": {"hour": 0.4, "day": 0.3,
                                            "week": 0.2, "month": 0.1}}},
           "train": {"total_steps": 12}}
    ds = _FakeBandDS()
    full = list(iter(BandMixSampler(ds, 64, 42, 1, cfg)))
    assert len(full) == 12
    for skip_n in (1, 5, 11):
        skipped = list(iter(BandMixSampler(ds, 64, 42, 1, cfg,
                                           skip_micro=skip_n)))
        _check(f"band_mix skip={skip_n}", full, skipped, skip_n)
    # 越界保护：skip > 总步数 → 一个 batch 都不产出（而不是抛异常 ✓）
    assert list(iter(BandMixSampler(ds, 64, 42, 1, cfg, skip_micro=999))) == []


def test_shard_skip() -> None:
    print("[2/3] shard sampler 跳批对齐")
    ds = _FakeShardDS()
    full = list(iter(ShardBatchSampler(ds, 64, 42)))
    assert len(full) == 64, f"期望 ceil(4096/64)=64 批，实得 {len(full)}"
    for skip_n in (1, 7, len(full) - 1):
        skipped = list(iter(ShardBatchSampler(ds, 64, 42,
                                             skip_batches=skip_n)))
        _check(f"shard skip={skip_n}", full, skipped, skip_n)


def test_rng_roundtrip() -> None:
    """随机源 round-trip：存下来 → 另外消耗一批 → 还原 → 下一个抽样必须一致 ✓。"""
    print("[3/3] 随机源 round-trip（python / numpy / torch）")
    py, npg = random.getstate(), np.random.get_state()
    tch = torch.get_rng_state()
    ref = (random.random(), float(np.random.random()),
           float(torch.rand(1)))
    random.setstate(py)
    np.random.set_state(npg)
    torch.set_rng_state(tch)
    again = (random.random(), float(np.random.random()),
             float(torch.rand(1)))
    assert ref == again, f"随机源还原后抽样不一致 ✗ {ref} vs {again}"
    print(f"  ✓ 三路随机源 round-trip 一致：{ref[0]:.6f} / {ref[1]:.6f} / "
          f"{ref[2]:.6f}")


def main() -> None:
    test_band_mix_skip()
    test_shard_skip()
    test_rng_roundtrip()
    print("[OK] 断点续跑的数据流对齐门通过 ✓")


if __name__ == "__main__":
    main()
