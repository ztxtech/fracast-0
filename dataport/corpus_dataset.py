"""CorpusDataset：新语料（连续 mmap + 全局 offsets）的训练端读取器。

格式出处：写出端 `dataport/build_corpus.py`、流程 `pipeline/build_corpus.py` ✓。
每个 part 目录 `p<k>/` 里（转换完才有 index.npz / manifest.json ✓）：

    shardNNNN.values.f32.npy   float32 连续数据区（所有序列首尾相接）
    shardNNNN.offsets.npy      int64 N+1 → 第 i 条 = values[o[i]:o[i+1]]（O(1) ✓）
    shardNNNN.lengths.npy      int32 N
    shardNNNN.freq_id.npy      int16 N（**part 内局部**频率编号 → 本文件重映射 ✓）
    shardNNNN.ds_id.npy        int16 N（part 内局部数据集编号）
    shardNNNN.ts.npy           int64 N（起始时间戳，日历特征用 ✓）
    index.npz / manifest.json  行数、点数、逐分片行数（构造行→分片映射用 ✓）
    freqs.txt / datasets.txt   局部编号 → 名字

与 `ShardDataset` 的关系（PLAN #7 已定 ✓）：**子类化，只覆盖两处** ——
  ① `__init__`：索引改从 `index.npz` + `manifest.json` 构造；
  ② `_load_row`：用 `offsets` 做 O(1) 变长取行（**不重排数据** ✓），`valid = isfinite(values)`。
窗口构造 / 金字塔 / 归一化 / target / SNaive 锚全部继承父类
→ 与旧分片语料走**同一条语义** ✓（不会训练和评测两套口径 ✗）。

约定：本文件没有 CLI ✗（主入口只有根目录 `main.py` ✓）；
训练侧统一从 `dataport/dataport.py::build_train_loaders(cfg)` 进来 ✓。
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from dataport.shard_dataset import ShardDataset


def _part_key(p: Path) -> tuple[int, str]:
    """p<k> → 按 k 数值排序（p2 要排在 p10 前面 ✓）。"""
    digits = p.name[1:]
    return (int(digits) if digits.isdigit() else 1 << 30, p.name)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def scan_corpus(roots, verbose: bool = True, dataset_include=None) -> dict:
    """扫语料根（可多个）→ 训练索引；part 内的局部编号在这里统一重映射 ✓。

    · 只认**已写完**的 part（有 `p*/index.npz` ✓）；没写完的（转换中断）跳过并打印 ✗，
      不因为一个 part 没跑完就拖死整条训练管线 ✓。
    · `manifest.json` 的逐分片行数用来把「全局行号」映射到（分片, 分片内行号），
      并当场校验 `Σ 分片行数 == index.npz 行数` ✓（不一致早失败，别拿错位数据训练 ✗）。
    """
    roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(r) for r in roots]
    stems: list[str] = []
    freqs: list[str] = []
    freq_map: dict[str, int] = {}
    datasets: list[str] = []
    ds_map: dict[tuple[str, str], int] = {}
    shard_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    freq_ids: list[np.ndarray] = []
    ds_ids: list[np.ndarray] = []
    n_rows = n_points = n_shards = n_parts = 0
    skipped: list[str] = []

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"语料目录不存在: {root}")
        all_parts = sorted((p for p in root.glob("p*") if p.is_dir()), key=_part_key)
        done = [p for p in all_parts if (p / "index.npz").exists()]
        skipped.extend(f"{root.name}/{p.name}" for p in all_parts if p not in done)
        for part in done:
            manifest = json.loads((part / "manifest.json").read_text(encoding="utf-8"))
            with np.load(part / "index.npz", allow_pickle=False) as z:
                lengths = z["lengths"]
                part_freq = z["freq_id"]
                part_ds = z["ds_id"]
            counts = np.asarray([int(r) for _, r, _ in manifest["shards"]], dtype=np.int64)
            if int(counts.sum()) != int(lengths.size):
                raise ValueError(
                    f"{part}: manifest 声明 {int(counts.sum())} 行 ≠ index.npz {lengths.size} 行 "
                    f"—— 该 part 的转换可能中断了 ✗（数据不自洽不许进训练）")
            if lengths.size == 0:
                continue
            # part 内局部频率编号 → 全局编号（局部编号只在本 part 有意义 ✓）
            local = _read_lines(part / "freqs.txt")
            if local and int(part_freq.max()) >= len(local):
                raise ValueError(f"{part}: freq_id 到 {int(part_freq.max())}，"
                                 f"但 freqs.txt 只有 {len(local)} 条 ✗")
            remap = np.zeros(max(len(local), 1), dtype=np.int16)
            for i, name in enumerate(local):
                if name not in freq_map:
                    freq_map[name] = len(freqs)
                    freqs.append(name)
                remap[i] = freq_map[name]

            local_ds = _read_lines(part / "datasets.txt")
            if local_ds and int(part_ds.max()) >= len(local_ds):
                raise ValueError(f"{part}: ds_id 到 {int(part_ds.max())}，"
                                 f"但 datasets.txt 只有 {len(local_ds)} 条 ✗")
            ds_remap = np.zeros(max(len(local_ds), 1), dtype=np.int32)
            for i, name in enumerate(local_ds):
                key = (root.name, name)
                if key not in ds_map:
                    ds_map[key] = len(datasets)
                    datasets.append(f"{root.name}/{name}")
                ds_remap[i] = ds_map[key]

            # 数据集白名单（2026-09-15 用户定：只保留 TinyCast 实际用到的语料 ✓）。
            # 两种写法（本地名字见各 part 的 `datasets.txt` ✓）：
            #   "pret"                    → **整个 root** 全要（Pretrain 的 152 个数据集全保留）
            #   "chronos/training_corpus" → 只保留该 root 下的这一个数据集（KernelSynth）
            part_keep = None
            if dataset_include is not None:
                inc = set(dataset_include)
                if root.name not in inc:
                    keep_local = np.array(
                        [i for i, nm in enumerate(local_ds)
                         if f"{root.name}/{nm}" in inc], dtype=np.int64)
                    if keep_local.size == 0:
                        continue    # 本 part 一行都不要 → 整片跳过（不建索引、不 mmap ✓）
                    part_keep = np.isin(part_ds, keep_local)
            base = len(stems)
            starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
            sid = np.repeat(np.arange(base, base + counts.size), counts)
            rid = (np.arange(int(counts.sum()), dtype=np.int64)
                   - np.repeat(starts, counts))
            fid = remap[part_freq]
            did = ds_remap[part_ds]
            n_keep = int(counts.sum())
            if part_keep is not None:
                sid, rid = sid[part_keep], rid[part_keep]
                fid, did = fid[part_keep], did[part_keep]
                lengths = lengths[part_keep]
                n_keep = int(part_keep.sum())
            shard_ids.append(sid)
            row_ids.append(rid)
            freq_ids.append(fid)
            ds_ids.append(did)
            for k in range(counts.size):
                if not (part / f"shard{k:04d}.values.f32.npy").exists():
                    raise FileNotFoundError(
                        f"{part}: 缺 shard{k:04d}.values.f32.npy ✗")
                stems.append(str(part / f"shard{k:04d}"))
            n_shards += int(counts.size)
            n_rows += n_keep
            n_points += int(lengths.sum())
            n_parts += 1

    if not stems:
        raise FileNotFoundError(
            f"{[str(r) for r in roots]} 下没有已完成的 part（需要 p*/index.npz）—— "
            f"先跑 `main.py config/corpus/<名>.yaml`，再跑 `script/corpus/merge_corpus_parts.py` ✓")
    if verbose and skipped:
        print(f"[corpus] 跳过未完成的 part（无 index.npz）: {', '.join(skipped)}", flush=True)

    return {
        "roots": [str(r) for r in roots],
        "stems": stems,
        # int32 够用（分片数 ≪ 2^31、单分片行数 ≪ 2^31）且比 int64 省一半内存 ✓
        "shard_ids": np.concatenate(shard_ids).astype(np.int32),
        "row_ids": np.concatenate(row_ids).astype(np.int32),
        "freq_ids": np.concatenate(freq_ids).astype(np.int16),
        "ds_ids": np.concatenate(ds_ids).astype(np.int32),
        "freqs": freqs,
        "datasets": datasets,
        "n_parts": n_parts,
        "n_shards": n_shards,
        "n_rows": n_rows,
        "n_points": n_points,
        "skipped": skipped,
    }


class CorpusDataset(ShardDataset):
    """新语料训练端：连续 mmap + 全局 offsets；窗口语义继承 `ShardDataset` ✓。"""

    def __init__(self, corpus_dirs, cfg, train: bool = True, cache_shards: int = 64):
        self.dir = Path(corpus_dirs[0] if isinstance(corpus_dirs, (list, tuple))
                        else corpus_dirs)
        idx = scan_corpus(
            corpus_dirs,
            dataset_include=(cfg.get("data") or {}).get("dataset_include"))
        self._shard_files = idx["stems"]
        self._freqs = idx["freqs"]
        keep = self._split_mask(len(idx["shard_ids"]), cfg, train)
        self._shard_ids = idx["shard_ids"][keep]
        self._row_ids = idx["row_ids"][keep]
        self._freq_ids = idx["freq_ids"][keep]
        self._ds_ids = idx["ds_ids"][keep]
        self._datasets = idx["datasets"]
        # 逐行季节 scale 侧车缓存（合成分片带官方 `scale_factors.f32` → 直接用真值，不靠 freq 推 ✓）
        self._scale_cache: dict[str, np.ndarray] = {}
        self._scale_missing: set[str] = set()

        self._init_windows(cfg, train, cache_shards)
        self.fast_layout = True      # 本布局恒为连续 mmap（`_load_row` 直接用 offsets ✓）
        self._ts_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.split = "train" if train else "val"
        print(f"[corpus] {'+'.join(Path(r).name for r in idx['roots'])}: "
              f"{idx['n_parts']} part / {idx['n_shards']} 分片 / {idx['n_rows']:,} 行 / "
              f"{idx['n_points'] / 1e9:.2f}B 点 / {len(self._datasets):,} 数据集 "
              f"→ {self.split} {len(self._shard_ids):,} 行",
              flush=True)

    def _load_row(self, shard_file: str, row: int
                  ) -> tuple[np.ndarray, np.ndarray, int]:
        """返回 (values[T], valid[T], ts0)：offsets O(1) 取行 ✓，valid=isfinite ✓。

        本布局**没有 valid 文件** ✓ —— 缺失由 NaN 承载（写出端保留 NaN），掩码在这里现算。
        """
        item = self._cache.get(shard_file)
        if item is None:
            item = {
                "values": np.load(f"{shard_file}.values.f32.npy", mmap_mode="r"),
                "offsets": np.load(f"{shard_file}.offsets.npy", mmap_mode="r"),
                "ts": np.load(f"{shard_file}.ts.npy", mmap_mode="r"),
            }
            self._cache[shard_file] = item
            while len(self._cache) > self._cache_shards:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(shard_file)
        o0 = int(item["offsets"][row])
        o1 = int(item["offsets"][row + 1])
        values = np.asarray(item["values"][o0:o1], dtype=np.float32)
        return values, np.isfinite(values), int(item["ts"][row])

    def _sidecar_scale(self, i: int):
        """逐行季节 scale 侧车：`<stem>.scale.f32.npy`（没有则 None → 回退 freq 推导 ✓）。"""
        stem = self._shard_files[int(self._shard_ids[i])]
        if stem in self._scale_missing:
            return None
        arr = self._scale_cache.get(stem)
        if arr is None:
            p = Path(f"{stem}.scale.f32.npy")
            if not p.exists():
                self._scale_missing.add(stem)
                return None
            arr = np.load(p, mmap_mode="r")
            self._scale_cache[stem] = arr
        r = int(self._row_ids[i])
        if r >= int(arr.shape[0]):
            return None
        v = float(arr[r])
        return v if np.isfinite(v) and v > 0 else None

    def _row_scale(self, row: int, freq: str) -> float:
        """合成分片有侧车就用官方的逐行 scale；否则回退 freq → seasonal_scale_factor ✓。"""
        s = self._sidecar_scale(int(row))
        if s is not None:
            return float(s)
        return super()._row_scale(row, freq)
