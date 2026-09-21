"""Convert official TinyCast synthetic shards to the fast corpus layout.

These shards carry a per-series seasonal scale sidecar in addition to series.f16.
The scale is required by the committing loss, so a generic Arrow conversion
would discard it. This converter writes the fast layout directly and stores
the official scale in shard0000.scale.f32.npy. The corpus reader prefers that
sidecar and falls back to frequency-derived seasonality only when it is absent.

series.f16 stores per-series standardized values. Dequantization is therefore
payload * series_stdev + series_mean, which must happen before training.

The synthetic windows use 144 steps per day, so they are labeled 10T with a
day-aligned start. Conversion is CPU- and IO-bound and creates one output part
per source shard.

Usage:
    env -u PYTHONPATH .venv/bin/python script/corpus/build_synth_corpus.py \
        --src data/tinycast_synth_hf --out data/corpus_fast/tinycast_synth
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus_parts.py \
        --corpus tinycast_synth=data/corpus_fast/tinycast_synth:4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DAY = 86400
FREQ = "10T"  # 144 steps per day.
T0 = (1_700_000_000 // DAY) * DAY  # Day-aligned start timestamp.
BUCKET_EDGES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 30]


def read_source(d: Path) -> dict:
    """Validate and read one official synthetic shard without writing output."""
    need = ["series.f16", "offsets.npy", "lengths.npy", "scale_factors.f32",
            "series_mean.f32", "series_stdev.f32", "dataset_id.u16",
            "dataset_names.json"]
    for name in need:
        if not (d / name).exists():
            raise FileNotFoundError(f"{d} is missing {name}")
    lens = np.load(d / "lengths.npy").astype(np.int64)
    offsets = np.load(d / "offsets.npy").astype(np.int64)  # Byte offsets from the source.
    # dataset_id.u16 and scale/mean/std files are raw arrays rather than .npy files.
    ds = np.fromfile(d / "dataset_id.u16", dtype=np.uint16).astype(np.int16)
    sf = np.fromfile(d / "scale_factors.f32", dtype=np.float32)
    mean = np.fromfile(d / "series_mean.f32", dtype=np.float32)
    std = np.fromfile(d / "series_stdev.f32", dtype=np.float32)
    names = json.loads((d / "dataset_names.json").read_text(encoding="utf-8"))["datasets"]
    n = int(lens.size)
    # Source offsets are per-series starting byte positions without a sentinel; the
    # final length supplies the endpoint.
    if offsets.shape[0] != n:
        raise ValueError(f"{d}/offsets.npy has {offsets.shape[0]} rows, expected {n}")
    for nm, arr in (("dataset_id.u16", ds), ("scale_factors.f32", sf),
                    ("series_mean.f32", mean), ("series_stdev.f32", std)):
        if arr.shape[0] != n:
            raise ValueError(f"{d}/{nm} has {arr.shape[0]} rows, expected {n}")
    if int(offsets[0]) != 0 or not np.array_equal(np.diff(offsets), lens[:-1] * 2):
        raise ValueError(f"{d}: offsets and lengths disagree for float16 payload")
    if int(ds.max(initial=0)) >= len(names):
        raise ValueError(f"{d}: dataset_id exceeds names length {len(names)}")
    size = (d / "series.f16").stat().st_size
    end = int(offsets[-1]) + int(lens[-1]) * 2
    if size != end:
        raise ValueError(f"{d}: series.f16 has {size} bytes, expected {end}")
    return {"lens": lens, "offsets": offsets, "ds": ds, "sf": sf,
            "mean": mean, "std": std, "names": list(names), "n": n}


def convert_shard(src: Path, out: Path, part: int, parts: int,
                  chunk_rows: int = 2048, dry: bool = False) -> dict:
    """Convert one official synthetic shard into one corpus part."""
    meta = read_source(src)
    lens, ds, names = meta["lens"], meta["ds"], meta["names"]
    total = int(lens.sum())
    if dry:
        return {"part": part, "rows": meta["n"], "points": total,
                "datasets": names, "src": str(src)}
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "shard0000"
    series = np.memmap(src / "series.f16", dtype=np.float16, mode="r")
    vals = np.lib.format.open_memmap(f"{stem}.values.f32.npy", mode="w+",
                                     dtype=np.float32, shape=(total,))
    base = 0
    t0 = time.time()
    for c0 in range(0, meta["n"], chunk_rows):
        c1 = min(meta["n"], c0 + chunk_rows)
        rows = []
        for j in range(c0, c1):
            b = int(meta["offsets"][j]) // 2  # Convert float16 element offsets.
            z = np.asarray(series[b:b + int(lens[j])], dtype=np.float32)
            rows.append(z * float(meta["std"][j]) + float(meta["mean"][j]))
        seg = np.concatenate(rows)
        vals[base:base + seg.size] = seg
        base += seg.size
        if c0 % (chunk_rows * 16) == 0:
            print(f"    {c1:>7d}/{meta['n']:,} rows  [{time.time()-t0:.0f}s]", flush=True)
    vals.flush()
    del vals
    # Write point offsets, matching the fast-corpus reader contract.
    np.save(f"{stem}.offsets.npy", np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))
    np.save(f"{stem}.lengths.npy", lens.astype(np.int32))
    np.save(f"{stem}.freq_id.npy", np.zeros(meta["n"], dtype=np.int16))
    np.save(f"{stem}.ds_id.npy", ds.astype(np.int16))
    np.save(f"{stem}.ts.npy", np.full(meta["n"], T0, dtype=np.int64))
    np.save(f"{stem}.scale.f32.npy", meta["sf"].astype(np.float32))
    (out / "freqs.txt").write_text(FREQ + "\n", encoding="utf-8")
    (out / "datasets.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "ds_names.json").write_text(json.dumps(
        {"part": part, "parts": parts, "src": str(src),
         "datasets": {str(i): d for i, d in enumerate(names)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    buckets = {f"le{int(e)}": np.flatnonzero(lens <= e).astype(np.int64)
               for e in BUCKET_EDGES}
    np.savez(out / "index.npz", lengths=lens.astype(np.int32),
             freq_id=np.zeros(meta["n"], dtype=np.int16), ds_id=ds.astype(np.int16),
             **buckets)
    (out / "manifest.json").write_text(json.dumps(
        {"n_shards": 1, "n_rows": meta["n"], "n_points": total,
         "shards": [[0, meta["n"], total]], "n_datasets": len(names),
         "src": str(src), "part": part, "parts": parts, "file_mod": [0, 1],
         "datasets": names,
         "notes": "Direct TinyCast synthetic conversion; values are dequantized float32, "
                  "offsets are point offsets, and scale.f32.npy is the official per-series sidecar."},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return {"part": part, "rows": meta["n"], "points": total,
            "datasets": names, "src": str(src)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root containing shard_*/series.f16")
    ap.add_argument("--out", required=True, help="Output root; p0..pN parts are created")
    ap.add_argument("--dry", action="store_true", help="Validate and summarize without writing")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    shards = sorted(p for p in src.glob("shard_*") if (p / "series.f16").exists())
    if not shards:
        raise SystemExit(f"no shard_*/series.f16 inputs under {src}")
    print(f"[synth] {len(shards)} source shards -> {out}; freq={FREQ}, start={T0}",
          flush=True)
    n_rows = n_pts = 0
    for k, sd in enumerate(shards):
        print(f"  == {sd.name} -> p{k}", flush=True)
        s = convert_shard(sd, out / f"p{k}", k, len(shards), dry=a.dry)
        n_rows += s["rows"]
        n_pts += s["points"]
        print(f"     PASS {s['rows']:,} rows / {s['points']/1e9:.3f}B points", flush=True)
    suffix = " (dry run)" if a.dry else ""
    print(f"[synth] PASS {len(shards)} parts / {n_rows:,} rows / {n_pts/1e9:.3f}B points / "
          f"{n_pts*4/1e9:.2f} GB{suffix}", flush=True)

if __name__ == "__main__":
    main()
