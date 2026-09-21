"""Corpus-part rotation sampler for training.

Rotation is a pipeline policy rather than a model property. Each chunk reads
from one corpus part for ``chunk_steps`` optimization steps. Rows within a part
are shuffled normally.

When ``coverage`` is enabled, each part first emits a coverage batch containing
at least one sample from every observed dataset-frequency group. The remaining
rows are then consumed in shuffled batches. This keeps small datasets visible
when chunks are short.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def make_rotation_sampler(dataset, batch_size: int, seed: int,
                          grad_accum_steps: int, cfg: dict,
                          skip_micro: int = 0):
    """Construct a rotation batch sampler for ``CorpusDataset``."""
    policy = (cfg.get("train") or {}).get("sampling") or {}
    rotation = policy.get("rotation") or {}
    accum = max(1, int(grad_accum_steps))
    return RotationBatchSampler(
        dataset,
        batch_size,
        seed,
        chunk_steps=int(rotation["chunk_steps"]) * accum,
        total_steps=(int((cfg.get("train") or {}).get("total_steps") or 0)
                     * accum),
        grad_accum_steps=accum,
        order=str(rotation.get("order", "shuffled")),
        coverage=bool(rotation.get("coverage", False)),
        skip_micro=skip_micro,
    )


class RotationBatchSampler:
    """Yield micro batches while rotating through corpus parts."""

    def __init__(self, dataset, batch_size: int, seed: int, chunk_steps: int,
                 total_steps: int, grad_accum_steps: int = 1,
                 order: str = "shuffled", coverage: bool = False,
                 skip_micro: int = 0):
        if chunk_steps <= 0:
            raise ValueError("rotation.chunk_steps must be positive")
        if order not in ("round_robin", "shuffled"):
            raise ValueError(f"Invalid rotation.order: {order}")
        accum = max(1, int(grad_accum_steps))
        if chunk_steps % accum or total_steps % accum:
            raise ValueError(
                "Rotation micro steps must be divisible by grad_accum_steps")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.chunk_steps = int(chunk_steps)
        self.total_steps = int(total_steps)
        self.grad_accum_steps = accum
        self.order = order
        self.epoch = 0
        self.coverage = bool(coverage)
        self.skip_micro = max(0, min(int(skip_micro), self.total_steps))

        shard_ids = np.asarray(dataset._shard_ids, dtype=np.int64)
        parents = np.asarray([
            str(Path(str(path)).parent) for path in dataset._shard_files
        ])
        names, file_part = np.unique(parents, return_inverse=True)
        part_ids = file_part[shard_ids]
        self.order_idx = np.argsort(part_ids, kind="stable")
        counts = np.bincount(part_ids, minlength=len(names))
        self.bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.parts = [Path(name) for name in names]

        self.group_ids = None
        if coverage:
            n_freq = max(1, len(dataset._freqs))
            self.group_ids = (
                np.asarray(dataset._ds_ids, dtype=np.int64) * n_freq
                + np.asarray(dataset._freq_ids, dtype=np.int64)
            )
        print(
            f"[rotation] {len(names)} corpus parts / "
            f"chunk={self.chunk_steps // accum} optimization steps "
            f"({self.chunk_steps} micro steps) / batch={self.batch_size} "
            f"x accum={accum} / order={self.order} / coverage={self.coverage}",
            flush=True,
        )

    def __len__(self) -> int:
        return self.total_steps if self.total_steps > 0 else math.ceil(
            len(self.order_idx) / self.batch_size)

    def _part_batches(self, part: int, rng: np.random.Generator):
        start, end = int(self.bounds[part]), int(self.bounds[int(part) + 1])
        positions = self.order_idx[start:end].copy()
        if self.coverage:
            groups = self.group_ids[positions]
            order = rng.permutation(positions.size)
            _, first = np.unique(groups[order], return_index=True)
            selected = order[first]
            coverage_positions = positions[selected]
            rng.shuffle(coverage_positions)
            for offset in range(0, len(coverage_positions), self.batch_size):
                yield coverage_positions[offset:offset + self.batch_size].tolist()

            coverage_mask = np.zeros(positions.size, dtype=bool)
            coverage_mask[selected] = True
            positions = positions[~coverage_mask]

        while True:
            rng.shuffle(positions)
            for offset in range(0, len(positions), self.batch_size):
                batch = positions[offset:offset + self.batch_size].tolist()
                rng.shuffle(batch)
                yield batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        parts = np.arange(len(self.parts))
        if self.order == "shuffled":
            rng.shuffle(parts)
        remaining = self.total_steps
        part_iterators = [self._part_batches(int(part), rng) for part in parts]
        skip = self.skip_micro
        while remaining > 0:
            for iterator in part_iterators:
                if remaining <= 0:
                    break
                for _ in range(min(self.chunk_steps, remaining)):
                    batch = next(iterator)
                    remaining -= 1
                    if skip > 0:
                        skip -= 1
                        continue
                    yield batch
