"""把并行转换的多个 part 合并成单一全局索引（并做交叉核对）。

## 背景
转换是 32 路并行（`--part k --parts 32`），每个 part 写在 `corpus/p<k>/` 下，
各自有 `shard*.{values.f32,offsets,lengths,freq_id,ds_id}.npy` 与 `manifest.json`。
模型侧要的是**一个全局视图**：任意全局行号 → (分片, 片内行号)。

## 产出（都写在 corpus 根下）
- `index.npz`：`lengths` / `freq_id` / `ds_id` 全局拼接 + 长度分桶 `le*`（只建索引，不动数据）
- `shard_map.json`：全局分片序号 → 相对路径（含该分片的行数、点数）
- `datasets.txt` / `freqs.txt`：全局 id ↔ 名称（按 part 重映射，保证一致）
- `manifest.json`：n_shards / n_rows / n_points + 各 part 的声明值（用于交叉核对）

## 交叉核对（合并时顺手做）
1. **各 part manifest 声明之和** vs **实际重算**（读 lengths/offsets）→ 必须一致
2. 每个分片 `offsets[-1] == lengths.sum()` 且 `offsets[-1]*4 == 数据区字节`
3. 输出差异清单（任何不一致都要报，而不是静默通过）

## 用法
    env -u PYTHONPATH .venv/bin/python script/corpus/merge_corpus_parts.py --corpus data/corpus_fast/chronos
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


_PART_NAME_CACHE: dict = {}


def _part_names(pdir: str, fname: str):
    """读某个 part 的 datasets.txt/freqs.txt（带缓存）。"""
    key = (pdir, fname)
    if key in _PART_NAME_CACHE:
        return _PART_NAME_CACHE[key]
    fp = os.path.join(pdir, fname)
    names = [l for l in open(fp).read().splitlines() if l.strip()] if os.path.exists(fp) else []
    _PART_NAME_CACHE[key] = names
    return names


def npy_data_bytes(path: str) -> int:
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        hlen = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - hlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pattern", default="p*/shard*.values.f32.npy")
    a = ap.parse_args()

    parts = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.corpus, "p*", "")))
    vals = sorted(glob.glob(os.path.join(a.corpus, a.pattern)))
    print(f"  part 目录 {len([p for p in parts if os.path.isdir(p)])} 个；values 分片 {len(vals)} 个")
    if not vals:
        print("  ✗ 没有可合并的分片"); return

    # 数据集名/频率名：各 part 可能有自己的编号，统一重映射为全局编号
    ds_all, fr_all = [], []
    for p in sorted({os.path.dirname(v) for v in vals}):
        for fn, acc in (("datasets.txt", ds_all), ("freqs.txt", fr_all)):
            fp = os.path.join(p, fn)
            if os.path.exists(fp):
                for line in open(fp).read().splitlines():
                    if line.strip() and line not in acc:
                        acc.append(line)
    ds_id = {n: i for i, n in enumerate(ds_all)}
    fr_id = {n: i for i, n in enumerate(fr_all)}
    print(f"  数据集 {len(ds_all)} 个 / 频率 {len(fr_all)} 个（全局重映射）")

    shard_map, Ls, Fs, Ds = [], [], [], []
    _seen_manifests = set()
    declared_rows = declared_pts = 0
    calc_rows = calc_pts = 0
    problems = []
    for i, vf in enumerate(vals):
        base = vf[:-len(".values.f32.npy")]
        try:
            L = np.load(base + ".lengths.npy")
            O = np.load(base + ".offsets.npy")
        except Exception as e:
            problems.append((os.path.relpath(base, a.corpus), f"缺 lengths/offsets: {type(e).__name__}"))
            continue
        rows = len(L)
        pts = int(L.sum())
        calc_rows += rows
        calc_pts += pts
        if len(O) - 1 != rows:
            problems.append((os.path.relpath(base, a.corpus), f"行数不符 offsets={len(O)-1} lengths={rows}"))
        elif int(O[-1]) != pts:
            problems.append((os.path.relpath(base, a.corpus), f"末偏移 {int(O[-1])} != 长度和 {pts}"))
        elif int(O[-1]) * 4 != npy_data_bytes(vf):
            problems.append((os.path.relpath(base, a.corpus), "末偏移*4 != 数据区字节"))
        # 频率/数据集 id 重映射（分片内是按 part 的局部 id 存的）
        fj = base + ".freq_id.npy"
        dj = base + ".ds_id.npy"
        lf = np.load(fj) if os.path.exists(fj) else np.zeros(rows, np.int16)
        ld = np.load(dj) if os.path.exists(dj) else np.zeros(rows, np.int16)
        # ★ 全局重映射（2026-09-13 修）：分片里的 id 是 **该 part 的局部编号**，
        #   直接用根 datasets.txt/freqs.txt 索引会映射到错误的数据集。
        #   这里按「本 part 的名字表」→「全局名字表」的重映射表逐值转换。
        pdir = os.path.dirname(vf)
        p_ds_names = _part_names(pdir, "datasets.txt")
        p_fr_names = _part_names(pdir, "freqs.txt")
        if p_ds_names:
            m = np.array([ds_id.get(n, 0) for n in p_ds_names], dtype=np.int16)
            ld = m[np.clip(ld.astype(np.int64), 0, len(m) - 1)] if len(m) else ld
        if p_fr_names:
            mf = np.array([fr_id.get(n, 0) for n in p_fr_names], dtype=np.int16)
            lf = mf[np.clip(lf.astype(np.int64), 0, len(mf) - 1)] if len(mf) else lf
        Ls.append(L); Fs.append(lf); Ds.append(ld)
        shard_map.append({"path": os.path.relpath(vf, a.corpus), "rows": rows, "points": pts})
        # ★ 每个 part 的 manifest 只读一次（2026-09-13 修）：原来在分片循环里读，
        #   → 同一 part 的总数被它名下每个分片各累加一次 → 声明值虚高 ~19×
        mp = os.path.join(os.path.dirname(vf), "manifest.json")
        if os.path.exists(mp) and mp not in _seen_manifests:
            _seen_manifests.add(mp)
            try:
                man = json.load(open(mp))
                declared_rows += int(man.get("n_rows", 0))
                declared_pts += int(man.get("n_points", 0))
            except Exception as e:
                problems.append((os.path.relpath(mp, a.corpus), f"manifest 读取失败 {type(e).__name__}"))

    Lg = np.concatenate(Ls) if Ls else np.zeros(0, np.int32)
    Fg = np.concatenate(Fs) if Fs else np.zeros(0, np.int16)
    Dg = np.concatenate(Ds) if Ds else np.zeros(0, np.int16)
    edges = [64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 30]
    buckets = {f"le{e}": np.where(Lg <= e)[0].astype(np.int64) for e in edges}
    np.savez(os.path.join(a.corpus, "index.npz"), lengths=Lg, freq_id=Fg, ds_id=Dg, **buckets)
    json.dump({"n_shards": len(shard_map), "n_rows": int(Lg.size), "n_points": int(Lg.sum()),
               "n_datasets": len(ds_all), "shards": shard_map,
               "cross_check": {"declared_rows": declared_rows, "calc_rows": calc_rows,
                               "declared_points": declared_pts, "calc_points": calc_pts},
               "notes": "values=float32 连续；offsets 给 O(1) 取行；无长度/上下文截断"},
              open(os.path.join(a.corpus, "manifest.json"), "w"), indent=1)
    with open(os.path.join(a.corpus, "datasets.txt"), "w") as fh:
        fh.write("\n".join(ds_all) + "\n")
    with open(os.path.join(a.corpus, "freqs.txt"), "w") as fh:
        fh.write("\n".join(fr_all) + "\n")

    print(f"  ✓ 合并：{len(shard_map)} 分片 / {Lg.size:,} 序列 / {Lg.sum()/1e9:.3f}B 点 / "
          f"{Lg.sum()*4/1e9:.2f} GB(f32)")
    print(f"  交叉核对：part manifest 声明 {declared_rows:,} 行 / {declared_pts:,} 点  ↔  "
          f"实际重算 {calc_rows:,} 行 / {calc_pts:,} 点  "
          f"{'✓ 一致' if (declared_rows == calc_rows and declared_pts == calc_pts) else '✗ 不一致'}")
    if problems:
        print(f"  ✗ 异常 {len(problems)} 处:")
        for nm, why in problems[:10]:
            print(f"      {nm}: {why}")
    else:
        print("  ✓ 所有分片 offsets/lengths/数据区字节 三者自洽")


if __name__ == "__main__":
    main()
