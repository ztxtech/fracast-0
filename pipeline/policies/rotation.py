"""训练采样策略：按 corpus part 分块轮转。

轮转是 Pipeline 策略，不是模型结构 ✓。一个 chunk 固定读一个 part；
`chunk_steps` 控制每块占用多少个训练 step。批内数据仍在 part 内随机洗牌。

`coverage=true` 时，每个 part 先构造覆盖批：part 内每个
「数据集 × 频率」组至少出一个样本，再混入普通随机批。这样缩短 chunk 后，
115 个 part 都会被访问，小数据集也不会因为随机洗牌被完全挤掉。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def make_rotation_sampler(dataset, batch_size: int, seed: int,
                          grad_accum_steps: int, cfg: dict,
                          skip_micro: int = 0):
    """给 CorpusDataset 构造轮转 batch sampler（`build_train_loaders` 工厂口 ✓）。"""
    policy = (cfg.get("train") or {}).get("sampling") or {}
    rot = policy.get("rotation") or {}
    accum = max(1, int(grad_accum_steps))
    return RotationBatchSampler(
        dataset, batch_size, seed,
        chunk_steps=int(rot["chunk_steps"]) * accum,
        total_steps=(int((cfg.get("train") or {}).get("total_steps") or 0)
                     * accum),
        grad_accum_steps=accum,
        order=str(rot.get("order", "shuffled")),
        coverage=bool(rot.get("coverage", False)),
        skip_micro=skip_micro)


class RotationBatchSampler:
    """按 corpus part 轮转产 micro batch；`__len__` = 优化步数 × 累积步数。"""

    def __init__(self, dataset, batch_size: int, seed: int, chunk_steps: int,
                 total_steps: int, grad_accum_steps: int = 1,
                 order: str = "shuffled", coverage: bool = False,
                 skip_micro: int = 0):
        if chunk_steps <= 0:
            raise ValueError("rotation.chunk_steps 必须大于 0")
        if order not in ("round_robin", "shuffled"):
            raise ValueError(f"rotation.order 非法: {order}")
        accum = max(1, int(grad_accum_steps))
        if chunk_steps % accum or total_steps % accum:
            raise ValueError("轮转 micro 步数必须能被 grad_accum_steps 整除")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.chunk_steps = int(chunk_steps)
        self.total_steps = int(total_steps)
        self.grad_accum_steps = accum
        self.order = order
        self.epoch = 0
        self.coverage = bool(coverage)
        # 断点续跑：开头的 skip_micro 个 micro batch 照常从生成器里取（RNG 逐位
        # 对齐 ✓）但不产出，数据流精确接在断点上 ✓。
        self.skip_micro = max(0, min(int(skip_micro), self.total_steps))

        shard_ids = np.asarray(dataset._shard_ids, dtype=np.int64)
        # part 归属：先在分片维度算出目录，再散射到行；与旧实现对拍逐位相同。
        parents = np.asarray([str(Path(str(f)).parent)
                              for f in dataset._shard_files])
        names, file_part = np.unique(parents, return_inverse=True)
        part_ids = file_part[shard_ids]
        self.order_idx = np.argsort(part_ids, kind="stable")
        counts = np.bincount(part_ids, minlength=len(names))
        self.bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.parts = [Path(x) for x in names]

        # 覆盖批的组号 = 全局数据集 × 全局频率；元数据由 CorpusDataset 提供。
        self.group_ids = None
        if coverage:
            n_freq = max(1, len(dataset._freqs))
            self.group_ids = (
                np.asarray(dataset._ds_ids, dtype=np.int64) * n_freq
                + np.asarray(dataset._freq_ids, dtype=np.int64)
            )
        print(f"[rotation] {len(names)} corpus part / "
              f"chunk={self.chunk_steps // accum} 优化步 "
              f"({self.chunk_steps} micro) / batch={self.batch_size} × accum={accum} / "
              f"order={self.order} / coverage={self.coverage}", flush=True)

    def __len__(self) -> int:
        return self.total_steps if self.total_steps > 0 else \
            math.ceil(len(self.order_idx) / self.batch_size)

    def _part_batches(self, part: int, rng: np.random.Generator):
        a, b = int(self.bounds[part]), int(self.bounds[int(part) + 1])
        positions = self.order_idx[a:b].copy()
        if self.coverage:
            groups = self.group_ids[positions]
            order = rng.permutation(positions.size)
            _, first = np.unique(groups[order], return_index=True)
            selected = order[first]
            cover_positions = positions[selected]
            rng.shuffle(cover_positions)
            for s in range(0, len(cover_positions), self.batch_size):
                yield cover_positions[s:s + self.batch_size].tolist()

            cover_mask = np.zeros(positions.size, dtype=bool)
            cover_mask[selected] = True
            positions = positions[~cover_mask]

        while True:
            rng.shuffle(positions)
            for s in range(0, len(positions), self.batch_size):
                batch = positions[s:s + self.batch_size].tolist()
                rng.shuffle(batch)
                yield batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        parts = np.arange(len(self.parts))
        if self.order == "shuffled":
            rng.shuffle(parts)
        left = self.total_steps
        part_iters = [self._part_batches(int(p), rng) for p in parts]
        skip = self.skip_micro
        while left > 0:
            for it in part_iters:
                if left <= 0:
                    break
                for _ in range(min(self.chunk_steps, left)):
                    batch = next(it)
                    left -= 1
                    if skip > 0:
                        skip -= 1
                        continue      # 只推进生成器，不产出 batch（续跑对齐用 ✓）
                    yield batch
