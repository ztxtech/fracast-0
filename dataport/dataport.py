"""Data loading entry point for pretraining.

The training pipeline calls :func:`build_train_loaders` once.  The function
selects the configured corpus layout and returns aligned train and validation
loaders.
"""
from __future__ import annotations


def corpus_roots_of(cfg: dict):
    """Return the configured corpus roots as an ordered list."""
    data = cfg.get("data") or {}
    roots = data.get("corpus_dirs") or data.get("corpus_dir")
    if not roots:
        return None
    return list(roots) if isinstance(roots, (list, tuple)) else [roots]


def build_train_loaders(cfg: dict, sampler_factory=None, skip_micro: int = 0):
    """Build train and validation loaders for the configured corpus layout."""
    import torch
    from torch.utils.data import DataLoader

    from dataport.shard_dataset import ShardBatchSampler, collate_shard

    data = cfg.get("data") or {}
    roots = corpus_roots_of(cfg)
    if roots:
        from dataport.corpus_dataset import CorpusDataset as Dataset

        source = roots
    elif data.get("shard_dir"):
        from dataport.shard_dataset import ShardDataset as Dataset

        source = data["shard_dir"]
    else:
        raise ValueError(
            "data.corpus_dirs or data.shard_dir is required; "
            "run the corpus preparation flow first"
        )

    n_workers = int(data.get("num_workers", 8))
    batch_size = int(cfg["train"]["batch_size"])
    train_dataset = Dataset(source, cfg, train=True)
    val_dataset = Dataset(source, cfg, train=False)
    seed = int(data.get("seed", 42))
    grad_accum = max(1, int((cfg.get("train") or {}).get("grad_accum_steps", 1)))
    train_sampler = (
        sampler_factory(train_dataset, batch_size, seed, grad_accum)
        if sampler_factory is not None
        else ShardBatchSampler(
            train_dataset, batch_size, seed, skip_batches=skip_micro
        )
    )
    prefetch = int(data.get("prefetch_factor", 4)) if n_workers > 0 else None
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=n_workers,
        collate_fn=collate_shard,
        persistent_workers=n_workers > 0,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_shard,
        persistent_workers=True,
    )
    print(
        f"[data] {type(train_dataset).__name__}: "
        f"{len(train_dataset):,} train / {len(val_dataset):,} val "
        f"(batch={batch_size}, workers={n_workers})",
        flush=True,
    )
    return train_loader, val_loader, True
