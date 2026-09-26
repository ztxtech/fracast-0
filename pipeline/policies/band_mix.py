"""Frequency-band weighted batch sampling.

TinyCast reports a training mixture of 31% hourly, 30% sub-hourly, 15% daily,
8% weekly, 7% monthly, and 6% second-level data. The original training code was
not released, so this module implements that mixture for the Fracast corpus.

The sampler is designed for a corpus stored as mmap-backed shards. Drawing rows
uniformly from the complete corpus would touch hundreds of shards per batch. To
keep reads sequential, each batch selects one shard per frequency band and draws
a contiguous range of rows from that shard. The row marginal probability remains
uniform within each band; only the within-batch ordering is correlated.

Some bands may contain fewer rows than the target mixture implies. A band whose
single-shard capacity cannot fill its configured share is capped and the
remaining probability mass is redistributed across the other bands.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


def band_of(freq: str) -> str:
    """Map a frequency string to a broad sampling band."""
    value = str(freq).strip()
    if not value:
        return "other"
    if value in ("M", "MS", "ME", "BM", "BMS") or value.startswith("M-"):
        return "month"
    if value in ("A", "AS", "YS") or value.startswith("A-") or value.startswith("Y"):
        return "other"
    if value.startswith("Q"):
        return "other"
    if value.endswith("S"):
        return "second"
    if value.endswith("T") or value.lower().endswith("min"):
        number = int(value[:-1]) if value[:-1].isdigit() else 1
        return "subhour" if number < 60 else "hour"
    if value.endswith("H"):
        return "hour"
    if value.endswith("D"):
        return "day"
    if "W" in value:
        return "week"
    return "other"


def _band_shares(
    sizes: "OrderedDict[str, int]",
    requested: dict,
    batch_size: int,
) -> "OrderedDict[str, float]":
    """Normalize configured band shares and cap bands limited by their corpus."""
    named = {band: float(requested[band]) for band in sizes if band in requested}
    unnamed = [band for band in sizes if band not in requested]
    remaining = max(0.0, 1.0 - float(sum(named.values())))
    if unnamed:
        named.update({band: remaining / len(unnamed) for band in unnamed})
    keep = OrderedDict((band, share) for band, share in named.items() if share > 0)
    if not keep:
        raise ValueError("band_mix: all band shares are zero")

    capped: dict[str, float] = {}
    for _ in range(16):
        over = [
            band
            for band, share in keep.items()
            if share > sizes[band] / float(batch_size) + 1e-12
            and band not in capped
        ]
        if not over:
            break
        for band in over:
            capped[band] = keep[band]
            keep[band] = sizes[band] / float(batch_size)
        free = [band for band in keep if band not in capped and keep[band] > 0]
        left = max(0.0, 1.0 - float(sum(keep.values())))
        total = float(sum(keep[band] for band in free))
        if not free or total <= 0:
            break
        for band in free:
            keep[band] += left * keep[band] / total
    if capped:
        print(
            "[band] capped shares limited by available rows: "
            + " ".join(
                f"{band} {100 * share:.2f}%->{100 * keep[band]:.2f}%"
                for band, share in capped.items()
            ),
            flush=True,
        )

    total = float(sum(keep.values()))
    return OrderedDict((band, share / total) for band, share in keep.items())


class _BandPool:
    """Row pool for one frequency band, grouped by shard."""

    __slots__ = ("flat", "starts", "shards", "counts", "probabilities", "max_rows")

    def __init__(self, rows: np.ndarray, shard_ids: np.ndarray, n_shards: int):
        shard_of_row = shard_ids[rows]
        order = np.argsort(shard_of_row, kind="stable")
        self.flat = rows[order].astype(np.int32, copy=False)
        counts = np.bincount(shard_of_row, minlength=n_shards)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.shards = np.flatnonzero(counts).astype(np.int64)
        self.counts = counts[self.shards]
        self.probabilities = self.counts / float(self.counts.sum())
        self.max_rows = int(self.counts.max())

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a contiguous circular range from one weighted shard."""
        shard = int(self.shards[int(rng.choice(self.shards.size, p=self.probabilities))])
        start, end = int(self.starts[shard]), int(self.starts[shard + 1])
        available = end - start
        offset = int(rng.integers(0, available))
        positions = start + (offset + np.arange(int(count))) % available
        return self.flat[positions]


class BandMixSampler:
    """Batch sampler that follows the configured frequency mixture."""

    def __init__(self, dataset, batch_size: int, seed: int,
                 grad_accum_steps: int, cfg: dict, skip_micro: int = 0):
        data_cfg = cfg.get("data") or {}
        train_cfg = cfg.get("train") or {}
        mix = data_cfg.get("band_mix") or {}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.micro_steps = max(1, int(train_cfg.get("total_steps") or 1)) * max(
            1, int(grad_accum_steps)
        )
        self.skip_micro = max(0, min(int(skip_micro), self.micro_steps))

        codes = np.empty(max(len(dataset._freqs), 1), dtype=np.int8)
        code_of: "OrderedDict[str, int]" = OrderedDict()
        for index, frequency in enumerate(dataset._freqs):
            band = band_of(frequency)
            if band not in code_of:
                if len(code_of) > 100:
                    raise ValueError("band_mix: too many frequency bands")
                code_of[band] = len(code_of)
            codes[index] = code_of[band]
        band_codes = codes[np.asarray(dataset._freq_ids, dtype=np.int64)]

        shard_ids = np.asarray(dataset._shard_ids)
        n_shards = len(dataset._shard_files)
        pools: "OrderedDict[str, _BandPool]" = OrderedDict()
        for band, code in code_of.items():
            rows = np.flatnonzero(band_codes == code)
            if rows.size:
                pools[band] = _BandPool(rows, shard_ids, n_shards)
        if not pools:
            raise ValueError("band_mix: no rows are available for sampling")

        sizes = OrderedDict((band, pool.max_rows) for band, pool in pools.items())
        requested = {
            str(key): float(value)
            for key, value in (mix.get("shares") or {}).items()
        }
        shares = _band_shares(sizes, requested, self.batch_size)
        self.bands = list(shares.keys())
        self.probabilities = np.asarray(
            [shares[band] for band in self.bands], dtype=np.float64)
        self.pools = [pools[band] for band in self.bands]

        n_rows = int(sum(int(pool.flat.size) for pool in self.pools))
        print(
            f"[band] {len(self.bands)} bands / {n_rows:,} rows -> "
            + " ".join(
                f"{band}({pool.flat.size:,} rows,{100 * probability:.2f}%,"
                f"{pool.shards.size} shards)"
                for band, pool, probability in zip(
                    self.bands, self.pools, self.probabilities)
            ),
            flush=True,
        )

    def __len__(self) -> int:
        return self.micro_steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        need = np.floor(self.probabilities * self.batch_size).astype(np.int64)
        need[int(np.argmax(self.probabilities))] += (
            self.batch_size - int(need.sum())
        )
        for index in range(self.micro_steps):
            output = np.empty(self.batch_size, dtype=np.int32)
            offset = 0
            for band_index in range(len(self.bands)):
                count = int(need[band_index])
                if count > 0:
                    output[offset:offset + count] = self.pools[band_index].draw(
                        count, rng)
                    offset += count
            if index < self.skip_micro:
                continue
            yield output.tolist()


def make_band_mix_sampler(dataset, batch_size: int, seed: int,
                          grad_accum_steps: int, cfg: dict,
                          skip_micro: int = 0):
    """Construct a frequency-band sampler for the training loaders."""
    return BandMixSampler(
        dataset, batch_size, seed, grad_accum_steps, cfg, skip_micro=skip_micro)
