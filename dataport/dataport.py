"""DataPort: the single data-loading entry point for pretraining.

一条调用链，三种语料布局（选哪种只看配置 ✓，流程侧不关心 ✗）：

    main.py → Pipeline.train() → pipeline/train.py::run(config)
      └─ dataport/dataport.py::build_train_loaders(cfg)      ← 本文件（统一口 ✓）
           ├─ data.corpus_dirs  → CorpusDataset  fast mmap + global offsets
           └─ data.shard_dir    → ShardDataset   legacy compact shards

为什么要这一层：训练脚本里曾经直接 `if shard_dir: ... else: ...` 就地分支 ✗ ——
数据格式的细节漏进了流程层，格式一变就要改流程。现在流程只有**一行**调用 ✓。

返回 `(train_loader, val_loader, shard_mode)`，当前两条布局都返回 12 字段 batch。

约定：本文件没有 CLI ✗（主入口只有根目录 `main.py` ✓）。
"""
from __future__ import annotations


def corpus_roots_of(cfg: dict):
    """从配置取新语料根：`data.corpus_dirs`（可多个 ✓）或 `data.corpus_dir`（单个）。"""
    data = cfg.get("data") or {}
    roots = data.get("corpus_dirs") or data.get("corpus_dir")
    if not roots:
        return None
    return list(roots) if isinstance(roots, (list, tuple)) else [roots]


def build_train_loaders(cfg: dict, sampler_factory=None, skip_micro: int = 0):
    """按配置选语料布局并装配 train/val 两个 DataLoader（见模块 docstring ✓）。"""
    import torch
    from torch.utils.data import DataLoader

    from dataport.shard_dataset import ShardBatchSampler, collate_shard

    data = cfg.get("data") or {}
    roots = corpus_roots_of(cfg)
    if roots:
        from dataport.corpus_dataset import CorpusDataset as _Dataset

        source = roots                      # 新语料：连续 mmap + 全局 offsets ✓
    elif data.get("shard_dir"):
        from dataport.shard_dataset import ShardDataset as _Dataset

        source = data["shard_dir"]          # 旧分片：memmap/npz + 紧凑索引 ✓
    else:
        raise ValueError(
            "data.corpus_dirs or data.shard_dir is required; "
            "run the corpus preparation flow first"
        )

    n_workers = int(data.get("num_workers", 8))
    batch_size = int(cfg["train"]["batch_size"])
    tr_ds = _Dataset(source, cfg, train=True)
    va_ds = _Dataset(source, cfg, train=False)
    seed = int(data.get("seed", 42))
    grad_accum = max(1, int((cfg.get("train") or {}).get("grad_accum_steps", 1)))
    tr_sampler = (sampler_factory(tr_ds, batch_size, seed, grad_accum)
                  if sampler_factory is not None
                  else ShardBatchSampler(tr_ds, batch_size, seed,
                                         skip_batches=skip_micro))
    # 续跑时 sampler_factory 已经把 skip_micro 包进闭包（见 pipeline/train.py ✓）
    # prefetch_factor：每个 worker 预取的 batch 数（默认 4）——喂 GPU 的关键旋钮 ✓
    prefetch = int(data.get("prefetch_factor", 4)) if n_workers > 0 else None
    pin_memory = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, num_workers=n_workers,
                           collate_fn=collate_shard, persistent_workers=n_workers > 0,
                           pin_memory=pin_memory, prefetch_factor=prefetch)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=1,
                           collate_fn=collate_shard, persistent_workers=True)
    print(f"[data] {type(tr_ds).__name__}: {len(tr_ds):,} train / {len(va_ds):,} val "
          f"（batch={batch_size}, workers={n_workers}, shard_mode=True ✓）", flush=True)
    return tr_loader, va_loader, True
