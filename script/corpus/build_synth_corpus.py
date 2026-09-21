"""TinyCast 4 个合成分片 → `corpus_fast` 布局（**带官方逐序列 scale 侧车**）。

## 为什么不走通用转换（`config/corpus/*.yaml`）
官方合成片除了 `series.f16` 还带 **逐序列 `scale_factors.f32`**（`corpus.py::_sample_scale_factor`
按 log-uniform 0.05–6.0 抽，synthetic 序列没有采样频率，所以每个序列各带一个）。
这个 scale 是 committing 项里季节复制基准的 lag 来源
（`losses.seasonal_copy_baseline`: `lag = round(24 / scale)`），
而 arrow/parquet 契约里没有这一列 → 走通用转换会把它整片丢掉 ✗。
所以这里按官方分片**直接写出** fast 布局，并把 scale 落成 `<stem>.scale.f32.npy` 侧车；
读取器 `dataport/corpus_dataset.py::_sidecar_scale` 会优先用它，没有才回退 freq 推导 ✓。

## 反归一化（官方读回路径）
`series.f16` 是**逐序列 z 归一化后的载荷**，读回 = `payload * series_stdev + series_mean`
（`tinycast/corpus.py::normalize_for_f16` 的 docstring 明写）✓ —— 不反归一化会污染量纲 ✗。

## 频率与时间戳
这批片按**步数**生成（144 步 = 一天），所以标 `10T` + 日边界起点；
与 `script/corpus/convert_synth_to_arrow.py` 同一口径 ✓（该脚本产出的 arrow 在
`data/tinycast_synth_hf_arrow/`，本脚本不依赖它）。

## 跑法（CPU/IO 任务，不占 GPU；输出根按 source shard 一分为 `p0..p3`）
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
FREQ = "10T"                                   # 144 步/天 → 10 分钟
T0 = (1_700_000_000 // DAY) * DAY              # 日边界起点（同 arrow 转换口径 ✓）
BUCKET_EDGES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 30]


def read_source(d: Path) -> dict:
    """读一个官方合成片并做自洽校验（不通过就早失败，别拿错位数据落盘 ✗）。"""
    need = ["series.f16", "offsets.npy", "lengths.npy", "scale_factors.f32",
            "series_mean.f32", "series_stdev.f32", "dataset_id.u16",
            "dataset_names.json"]
    for name in need:
        if not (d / name).exists():
            raise FileNotFoundError(f"{d} 缺 {name} ✗")
    lens = np.load(d / "lengths.npy").astype(np.int64)
    offs = np.load(d / "offsets.npy").astype(np.int64)      # **字节**偏移 ✓
    # ★ `dataset_id.u16` 是**裸 uint16 载荷**（不是 .npy ✗）；offsets/lengths 才是 .npy ✓
    ds = np.fromfile(d / "dataset_id.u16", dtype=np.uint16).astype(np.int16)
    sf = np.fromfile(d / "scale_factors.f32", dtype=np.float32)
    mean = np.fromfile(d / "series_mean.f32", dtype=np.float32)
    std = np.fromfile(d / "series_stdev.f32", dtype=np.float32)
    names = json.loads((d / "dataset_names.json").read_text(encoding="utf-8"))["datasets"]
    n = int(lens.size)
    # ★ 官方片里 `offsets` 是**每条序列的起始字节**（n 条，无末尾哨兵 ✗），
    #   末条末尾要拿 lengths 单独推；`dataset_id.u16` 等是裸载荷（不是 .npy ✗）。
    if offs.shape[0] != n:
        raise ValueError(f"{d}/offsets.npy 行数 {offs.shape[0]} ≠ lengths {n} ✗")
    for nm, arr in (("dataset_id.u16", ds), ("scale_factors.f32", sf),
                    ("series_mean.f32", mean), ("series_stdev.f32", std)):
        if arr.shape[0] != n:
            raise ValueError(f"{d}/{nm} 行数 {arr.shape[0]} 与 lengths {n} 不一致 ✗")
    if int(offs[0]) != 0 or not np.array_equal(np.diff(offs), lens[:-1] * 2):
        raise ValueError(f"{d}: offsets 与 lengths 不自洽（f16=2 字节）✗")
    if int(ds.max(initial=0)) >= len(names):
        raise ValueError(f"{d}: dataset_id 越界（names={len(names)}）✗")
    size = (d / "series.f16").stat().st_size
    end = int(offs[-1]) + int(lens[-1]) * 2
    if size != end:
        raise ValueError(f"{d}: series.f16 有 {size} 字节 ≠ 末条末尾 {end} ✗")
    return {"lens": lens, "offsets": offs, "ds": ds, "sf": sf,
            "mean": mean, "std": std, "names": list(names), "n": n}


def convert_shard(src: Path, out: Path, part: int, parts: int,
                  chunk_rows: int = 2048, dry: bool = False) -> dict:
    """一个官方片 → 一个 part 目录（内含单个 `shard0000` ✓）。"""
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
            b = int(meta["offsets"][j]) // 2                  # f16 = 2 字节 ✓
            z = np.asarray(series[b:b + int(lens[j])], dtype=np.float32)
            rows.append(z * float(meta["std"][j]) + float(meta["mean"][j]))
        seg = np.concatenate(rows)
        vals[base:base + seg.size] = seg
        base += seg.size
        if c0 % (chunk_rows * 16) == 0:
            print(f"    {c1:>7d}/{meta['n']:,} 条  [{time.time()-t0:.0f}s]", flush=True)
    vals.flush()
    del vals
    # 点偏移（**不是字节**）：与 `dataport/build_corpus.py` 的契约一致 ✓
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
         "notes": "官方 TinyCast 合成片直写；values=反归一化后的 float32；"
                  "offsets 为点偏移；shard0000.scale.f32.npy = 官方逐序列 scale 侧车 ✓"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return {"part": part, "rows": meta["n"], "points": total,
            "datasets": names, "src": str(src)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 shard_*/series.f16 的官方合成片根目录")
    ap.add_argument("--out", required=True, help="输出根（自动建 p0..pN）")
    ap.add_argument("--dry", action="store_true", help="只校验与统计，不写盘")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    shards = sorted(p for p in src.glob("shard_*") if (p / "series.f16").exists())
    if not shards:
        raise SystemExit(f"✗ {src} 下没有 shard_*/series.f16")
    print(f"[synth] {len(shards)} 个官方合成片 → {out}（freq={FREQ}, 起点={T0}）",
          flush=True)
    n_rows = n_pts = 0
    for k, sd in enumerate(shards):
        print(f"  == {sd.name} → p{k}", flush=True)
        s = convert_shard(sd, out / f"p{k}", k, len(shards), dry=a.dry)
        n_rows += s["rows"]
        n_pts += s["points"]
        print(f"     ✓ {s['rows']:,} 条 / {s['points']/1e9:.3f}B 点", flush=True)
    print(f"[synth] ✓ {len(shards)} part / {n_rows:,} 条 / {n_pts/1e9:.3f}B 点 / "
          f"{n_pts*4/1e9:.2f} GB{f'（dry ✓）' if a.dry else ''}", flush=True)


if __name__ == "__main__":
    main()
