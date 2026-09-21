"""Dataset reader for the contiguous mmap corpus layout.

Each completed part contains one or more data shards, a global index, and a
manifest.  The reader remaps part-local frequency and dataset identifiers and
reuses the window construction implemented by ``ShardDataset``.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from dataport.shard_dataset import ShardDataset


def _part_key(p: Path) -> tuple[int, str]:
    """Sort part directories by numeric suffix rather than lexicographically."""
    digits = p.name[1:]
    return (int(digits) if digits.isdigit() else 1 << 30, p.name)


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def scan_corpus(roots, verbose: bool = True, dataset_include=None) -> dict:
    """Scan one or more corpus roots and build a global training index.

    Incomplete parts are skipped.  Shard row counts are validated against the
    global index before any data is exposed to training.
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
            raise FileNotFoundError(f"corpus directory does not exist: {root}")
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
                    f"{part}: manifest reports {int(counts.sum())} rows but "
                    f"index.npz contains {lengths.size}"
                )
            if lengths.size == 0:
                continue
            # Remap part-local frequency identifiers to global identifiers.
            local = _read_lines(part / "freqs.txt")
            if local and int(part_freq.max()) >= len(local):
                raise ValueError(
                    f"{part}: freq_id reaches {int(part_freq.max())} but "
                    f"freqs.txt contains {len(local)} entries"
                )
            remap = np.zeros(max(len(local), 1), dtype=np.int16)
            for i, name in enumerate(local):
                if name not in freq_map:
                    freq_map[name] = len(freqs)
                    freqs.append(name)
                remap[i] = freq_map[name]

            local_ds = _read_lines(part / "datasets.txt")
            if local_ds and int(part_ds.max()) >= len(local_ds):
                raise ValueError(
                    f"{part}: ds_id reaches {int(part_ds.max())} but "
                    f"datasets.txt contains {len(local_ds)} entries"
                )
            ds_remap = np.zeros(max(len(local_ds), 1), dtype=np.int32)
            for i, name in enumerate(local_ds):
                key = (root.name, name)
                if key not in ds_map:
                    ds_map[key] = len(datasets)
                    datasets.append(f"{root.name}/{name}")
                ds_remap[i] = ds_map[key]

            # Optional dataset allowlist.  A root name includes all datasets under
            # that root; ``root/dataset`` selects one dataset.
            part_keep = None
            if dataset_include is not None:
                inc = set(dataset_include)
                if root.name not in inc:
                    keep_local = np.array(
                        [i for i, nm in enumerate(local_ds)
                         if f"{root.name}/{nm}" in inc], dtype=np.int64)
                    if keep_local.size == 0:
                        continue
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
                        f"{part}: missing shard{k:04d}.values.f32.npy"
                    )
                stems.append(str(part / f"shard{k:04d}"))
            n_shards += int(counts.size)
            n_rows += n_keep
            n_points += int(lengths.sum())
            n_parts += 1

    if not stems:
        raise FileNotFoundError(
            f"no completed corpus parts found under {[str(r) for r in roots]}; "
            "run the corpus build flow and merge the parts first"
        )
    if verbose and skipped:
        print(f"[corpus] skipped incomplete parts: {', '.join(skipped)}", flush=True)

    return {
        "roots": [str(r) for r in roots],
        "stems": stems,
        # int32 is sufficient for shard and row counts and halves memory use.
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
    """Training dataset for contiguous mmap corpora with global offsets."""

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
        # Cache optional per-row seasonal scale sidecars.
        self._scale_cache: dict[str, np.ndarray] = {}
        self._scale_missing: set[str] = set()

        self._init_windows(cfg, train, cache_shards)
        self.fast_layout = True
        self._ts_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.split = "train" if train else "val"
        print(
            f"[corpus] {'+'.join(Path(r).name for r in idx['roots'])}: "
            f"{idx['n_parts']} parts / {idx['n_shards']} shards / "
            f"{idx['n_rows']:,} rows / {idx['n_points'] / 1e9:.2f}B points / "
            f"{len(self._datasets):,} datasets -> {self.split} "
            f"{len(self._shard_ids):,} rows",
            flush=True,
        )

    def _load_row(self, shard_file: str, row: int
                  ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(values, valid, start_time)`` for one row."""
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
        """Read an optional per-row scale sidecar."""
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
        """Use the sidecar scale when present; otherwise infer from frequency."""
        s = self._sidecar_scale(int(row))
        if s is not None:
            return float(s)
        return super()._row_scale(row, freq)
