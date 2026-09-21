"""语料 part 级复检：归属、缺口、分片自洽（**不信 manifest**，从落盘产物重算 ✓）。

## 为什么要有这个（2026-09-13 二次复检）
`corpus_report.py` 的「完成数」看的是旧日志 ✗，`merge_corpus_parts.py` 只看各 part 自报的
manifest ✗ —— 两者都发现不了「part 根本没跑出来」和「ds_id 与名字错位」。本脚本换一套独立判据：

1. **part 分配重算**：按源目录字节数重跑 LPT 装箱（`balance=bytes`，算法与 build_corpus 相同但
   **独立实现**）→ 与各 part 的 `datasets.txt` 逐项比（顺序敏感 ✓）。这是「数据集归属」bug 的根因检查。
2. **id 越界**：每个分片 `ds_id < 本 part 数据集数`、`freq_id < freqs 条数`（错位必然越界或对不上名）。
3. **分片自洽**：`offsets[-1] == lengths.sum()`、`len(offsets) == len(lengths)+1`、
   值区字节 == 4×点数、四个数组长度一致。
4. **缺口/残件**：`p0..p(N-1)` 缺哪个；有 shard 但没有 `manifest.json` 的算残件（单列，不算完成 ✗）。
5. `--write-ds-names`：给缺 `ds_names.json` 的 part 补一份（先过 1/2 项才写 ✓，避免固化错位映射）。

## 用法（项目根；远端一律 `env -u PYTHONPATH .venv/bin/python`）
    .venv/bin/python script/corpus/audit_corpus_parts.py --all
    .venv/bin/python script/corpus/audit_corpus_parts.py --all --write-ds-names
    .venv/bin/python script/corpus/audit_corpus_parts.py --corpus pret=data/corpus_fast/pret:32

产出：`tmp/corpus_report/parts_audit.json`（逐 part 明细）+ 终端表格 ✓。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tmp", "corpus_report")

# 公开预训练语料：名字 → (源目录, part 数, part 目录)
DEFAULT_CORPORA = {
    "pret": ("data/pretrain_full", 32, "data/corpus_fast/pret"),
    "lotsa": ("data/lotsa_full", 32, "data/corpus_fast/lotsa"),
    "chronos": ("data/chronos_full", 32, "data/corpus_fast/chronos"),
    "boom": ("data/boom", 8, "data/corpus_fast/boom"),
    "fev": ("data/fev", 4, "data/corpus_fast/fev"),
}

_PART_RE = re.compile(r"^part?(\d+)$")
# 数据集内并行的桶：`p0_split0..p0_split7` —— 同属 part 0（file_mod 切分 ✓）
_SPLIT_RE = re.compile(r"^p(\d+)_split\d+$")


def npy_data_bytes(path: str) -> int:
    """`.npy` 的数据区字节数（扣掉 magic + header ✓）。"""
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        hlen = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - hlen


def lpt_pack(src_abs: str, parts: int) -> list[list[str]]:
    """重算 `balance=bytes` 的 LPT 装箱（独立实现；只统计含 arrow/parquet 的数据集 ✓）。"""
    all_ds = []
    for d in sorted(os.listdir(src_abs)):
        p = os.path.join(src_abs, d)
        if not os.path.isdir(p) or d.startswith("."):
            continue
        has, tot = False, 0
        for root, _dd, files in os.walk(p):
            for fn in files:
                if fn.endswith((".arrow", ".parquet")):
                    has = True
                try:
                    tot += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        if has:
            all_ds.append((d, tot))
    size = dict(all_ds)
    buckets: list[list[str]] = [[] for _ in range(parts)]
    load = [0] * parts
    for d in sorted(size, key=lambda x: -size[x]):
        k = load.index(min(load))
        buckets[k].append(d)
        load[k] += size[d]
    return buckets


def read_names(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    return [l for l in open(path).read().splitlines() if l.strip()]


def check_part(pd: str, expected: list[str] | None) -> dict:
    """复检一个 part 目录 → 明细 dict ✓。"""
    name = os.path.basename(pd)
    datasets = read_names(os.path.join(pd, "datasets.txt"))
    freqs = read_names(os.path.join(pd, "freqs.txt"))
    shards = sorted(glob.glob(os.path.join(pd, "shard*.values.f32.npy")))
    bad: list[str] = []
    n_rows = n_pts = 0
    max_ds = max_freq = -1
    for vf in shards:
        base = vf[: -len(".values.f32.npy")]
        try:
            L = np.load(base + ".lengths.npy")
            O = np.load(base + ".offsets.npy")
            D = np.load(base + ".ds_id.npy")
            F = np.load(base + ".freq_id.npy")
            T = np.load(base + ".ts.npy")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{os.path.basename(vf)}: 数组读不出 {type(exc).__name__}")
            continue
        if O.size != L.size + 1:
            bad.append(f"{os.path.basename(vf)}: offsets 长度 {O.size} != lengths {L.size}+1")
        elif int(O[-1]) != int(L.sum()):
            bad.append(f"{os.path.basename(vf)}: offsets[-1]={int(O[-1])} != lengths.sum()={int(L.sum())}")
        if D.size != L.size or F.size != L.size or T.size != L.size:
            bad.append(f"{os.path.basename(vf)}: ds/freq/ts 长度与 lengths 不一致")
        if npy_data_bytes(vf) != 4 * int(L.sum()):
            bad.append(f"{os.path.basename(vf)}: 值区字节 {npy_data_bytes(vf)} != 4×{int(L.sum())}")
        n_rows += int(L.size)
        n_pts += int(L.sum())
        if D.size:
            max_ds = max(max_ds, int(D.max()))
        if F.size:
            max_freq = max(max_freq, int(F.max()))
    if datasets and max_ds >= len(datasets):
        bad.append(f"ds_id 越界：max={max_ds} 但 datasets.txt 只有 {len(datasets)} 条")
    if freqs and max_freq >= len(freqs):
        bad.append(f"freq_id 越界：max={max_freq} 但 freqs.txt 只有 {len(freqs)} 条")
    if expected is not None and sorted(datasets) != sorted(expected):
        bad.append("part 分配与重算的 LPT 装箱不一致")
    return {"part": name, "n_shards": len(shards), "n_rows": n_rows, "n_points": n_pts,
            "n_datasets": len(datasets), "has_manifest": os.path.exists(os.path.join(pd, "manifest.json")),
            "has_ds_names": os.path.exists(os.path.join(pd, "ds_names.json")),
            "datasets": datasets, "expected": expected, "problems": bad,
            "_dir": pd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", default=[],
                    help="名字=源目录:part 数（可重复）；不给就用 --all 的三套默认值")
    ap.add_argument("--all", action="store_true", help="复检 README 中的全部公开语料")
    ap.add_argument("--write-ds-names", action="store_true", help="给缺 ds_names.json 的 part 补写 ✓")
    a = ap.parse_args()

    specs = dict(DEFAULT_CORPORA)
    if a.corpus:
        specs = {}
        for item in a.corpus:
            name, rest = item.split("=", 1)
            src, parts = rest.rsplit(":", 1)
            specs[name] = (src, int(parts), f"data/corpus_fast/{name}")

    os.makedirs(OUT, exist_ok=True)
    report: dict = {}
    rc = 0
    for name, (src, parts, cdir) in specs.items():
        src_abs = os.path.join(ROOT, src)
        cdir_abs = os.path.join(ROOT, cdir)
        if not os.path.isdir(cdir_abs):
            print(f"[{name}] 语料目录不存在：{cdir}")
            continue
        pack = lpt_pack(src_abs, parts) if os.path.isdir(src_abs) else None
        if pack is None:
            print(f"[{name}] 源目录不存在（跳过分配复核）：{src}")
        dirs = [d for d in sorted(glob.glob(os.path.join(cdir_abs, "p*"))) if os.path.isdir(d)]
        entries = []
        seen_nums = set()
        n_split_dirs = 0
        for pd in dirs:
            bn = os.path.basename(pd)
            ms = _SPLIT_RE.fullmatch(bn)
            if ms is not None:
                # 切分桶：只做自洽检查，不比对装箱（装箱口径与主 part 相同 ✓）
                n_split_dirs += 1
                seen_nums.add(int(ms.group(1)))
                entries.append(check_part(pd, None))
                continue
            m = _PART_RE.fullmatch(bn)
            exp = None
            if m is not None and pack is not None:
                k = int(m.group(1))
                seen_nums.add(k)
                exp = pack[k] if k < len(pack) else None
            entries.append(check_part(pd, exp))
        missing = sorted(set(range(parts)) - seen_nums) if pack is not None else []
        # ★ 数据集覆盖：合并后的 datasets.txt ↔ 源目录清单（逐名比，缺一个都算 ✗）
        #   （2026-09-13 二次复检加：以前只看「part 数」，看不出某个数据集根本没进语料 ✗）
        cov = None
        merged_names = read_names(os.path.join(cdir_abs, "datasets.txt"))
        if merged_names and pack is not None:
            src_names = sorted({d for b in pack for d in b})
            cov = {"src": len(src_names), "merged": len(merged_names),
                   "missing": sorted(set(src_names) - set(merged_names)),
                   "extra": sorted(set(merged_names) - set(src_names))}
            if cov["missing"] or cov["extra"]:
                print(f"        [覆盖] 源 {cov['src']} vs 合并 {cov['merged']}："
                      f"缺 {cov['missing']} / 多 {cov['extra']}")
        done = [e for e in entries if e["has_manifest"] and not e["problems"]]
        partial = [e for e in entries if not e["has_manifest"]]
        broken = [e for e in entries if e["problems"]]
        tot_rows = sum(e["n_rows"] for e in entries)
        tot_pts = sum(e["n_points"] for e in entries)
        _extra = f"（含数据集内并行桶 {n_split_dirs} 个）" if n_split_dirs else ""
        print(f"[{name}] part {len(seen_nums)}/{parts} · 目录 {len(entries)}{_extra} · 完成 {len(done)} · 残件 {len(partial)} · "
              f"问题 {len(broken)} · 缺 part {missing if missing else '无'}")
        print(f"        合计 {tot_rows:,} 条 / {tot_pts/1e9:.2f}B 点")
        if cov is not None:
            _ok = "✓" if not (cov["missing"] or cov["extra"]) else "✗"
            print(f"        数据集覆盖：源 {cov['src']} / 合并 {cov['merged']} 个 {_ok}")
        for e in partial:
            print(f"        [残件] {e['part']}: {e['n_shards']} 分片 / {e['n_rows']:,} 条（无 manifest ✗）")
        for e in broken:
            for p in e["problems"]:
                print(f"        [问题] {e['part']}: {p}")
        if a.write_ds_names:
            for e in entries:
                if e["has_ds_names"] or not e["datasets"] or e["problems"]:
                    continue
                with open(os.path.join(e["_dir"], "ds_names.json"), "w") as fh:
                    json.dump({"part": e["part"], "datasets": e["datasets"]}, fh, ensure_ascii=False, indent=1)
                print(f"        [写入] {e['part']}/ds_names.json（{len(e['datasets'])} 个数据集）")
        report[name] = {"src": src, "parts": parts, "n_part_dirs": len(entries), "n_split_dirs": n_split_dirs, "missing": missing,
                        "coverage": cov,
                        "complete": len(done), "partial": len(partial), "broken": len(broken),
                        "n_rows": tot_rows, "n_points": tot_pts,
                        "entries": [{k: v for k, v in e.items()} for e in entries]}
        if missing or partial or broken:
            rc = 1
    with open(os.path.join(OUT, "parts_audit.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    bad = any(v["missing"] or v["partial"] or v["broken"] for v in report.values())
    print(f"\n明细 → {os.path.relpath(os.path.join(OUT, 'parts_audit.json'), ROOT)}"
          f"（{'有问题，看上面 ✗' if bad else '全部通过 ✓'}）")
    sys.exit(rc)


if __name__ == "__main__":
    main()
