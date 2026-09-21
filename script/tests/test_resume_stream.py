"""CPU tests proving that skipped batches preserve the uninterrupted stream.

Weights, optimizer state, and step count can be stored, but stream alignment is
the hard part of resume. These tests replay the same fake corpus twice and
require skipped stream i to equal uninterrupted stream N+i exactly. Otherwise
later training samples change while the loss curve can still look normal.

A skip advances the sampler RNG normally but suppresses emitted batches; it
does not rebuild a new stream from the first N batches.

Run with: env -u PYTHONPATH .venv/bin/python script/tests/test_resume_stream.py
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
    """Provide only the fields consumed by the band-mix sampler."""

    def __init__(self, n_rows: int = 4096, n_shards: int = 6):
        rng = np.random.default_rng(0)
        self._freqs = ["5T", "H", "D", "W", "M"]  # One contiguous block per band.
        self._freq_ids = np.repeat(np.arange(len(self._freqs)), n_rows // 5)
        self._shard_ids = rng.integers(0, n_shards, size=self._freq_ids.size)
        self._shard_files = [f"/fake/shard{i}.npy" for i in range(n_shards)]


class _FakeShardDS:
    """Provide only the fields consumed by the shard batch sampler."""

    def __init__(self, n_rows: int = 4096, n_shards: int = 6):
        rng = np.random.default_rng(1)
        self._shard_ids = rng.integers(0, n_shards, size=n_rows)
        self._shard_files = [f"/fake/shard{i}.npy" for i in range(n_shards)]
        self.fast_layout = False  # Keep the CPU test away from prefetch workers.


def _check(name: str, full: list, skipped: list, skip_n: int) -> None:
    assert len(skipped) == len(full) - skip_n, (
        f"{name}: skipped stream length {len(skipped)}; expected {len(full) - skip_n}")
    bad = [i for i, (a, b) in enumerate(zip(full[skip_n:], skipped)) if a != b]
    assert not bad, (f"{name}: batches {bad[:5]} differ from the original stream; "
                     "the skip did not align the RNG")
    print(f"  PASS {name}: skipped {skip_n}/{len(full)}; {len(skipped)} batches match")


def test_band_mix_skip() -> None:
    print("[1/3] band-mix sampler skip alignment")
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
    # An out-of-range skip yields no batches rather than raising an exception.
    assert list(iter(BandMixSampler(ds, 64, 42, 1, cfg, skip_micro=999))) == []


def test_shard_skip() -> None:
    print("[2/3] shard sampler skip alignment")
    ds = _FakeShardDS()
    full = list(iter(ShardBatchSampler(ds, 64, 42)))
    assert len(full) == 64, f"expected 64 batches, got {len(full)}"
    for skip_n in (1, 7, len(full) - 1):
        skipped = list(iter(ShardBatchSampler(ds, 64, 42,
                                             skip_batches=skip_n)))
        _check(f"shard skip={skip_n}", full, skipped, skip_n)


def test_rng_roundtrip() -> None:
    """Check that saved Python, NumPy, and PyTorch RNG states round-trip exactly."""
    print("[3/3] Python / NumPy / PyTorch RNG round trip")
    py, npg = random.getstate(), np.random.get_state()
    tch = torch.get_rng_state()
    ref = (random.random(), float(np.random.random()),
           float(torch.rand(1)))
    random.setstate(py)
    np.random.set_state(npg)
    torch.set_rng_state(tch)
    again = (random.random(), float(np.random.random()),
             float(torch.rand(1)))
    assert ref == again, f"RNG restore changed the sample: {ref} vs {again}"
    print(f"  PASS three-stream RNG round trip: {ref[0]:.6f} / {ref[1]:.6f} / "
          f"{ref[2]:.6f}")


def main() -> None:
    test_band_mix_skip()
    test_shard_skip()
    test_rng_roundtrip()
    print("[OK] resumed data streams align with uninterrupted runs")


if __name__ == "__main__":
    main()
