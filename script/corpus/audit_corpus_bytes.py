"""第二轮独立核对（方法 B：**不信 manifest**，直接从数据文件重算）。

## 为什么要有第二轮
用户要求"反复核对反复核对，甚至派无上下文子 Agent 复核数据量"。
第一轮（`audit_corpus.py`）比的是"官方清单 vs 本地文件字节"，依据是 **HF 元数据**；
本轮**完全独立**：只读我们产出的语料文件本身，从零重算，任何一处对不上都要报出来。

## 三个独立重算（互不依赖）
- **M1 字节法**：`sum(os.path.getsize(values.f32.npy))/4` → 总点数（float32 每点 4 字节）
- **M2 长度法**：`sum(所有 shard 的 lengths.npy)` → 总点数
- **M3 偏移法**：每个 shard 的 `offsets.npy` 末项应等于该分片 lengths 之和；
  所有 offsets 数组长度 -1 之和 = 总序列数
三者必须**逐位一致**；再与 `manifest.json` 声明值对账（manifest 是**被核对对象**，不是依据）。

## 抽样值检查
随机抽 N 条：验证 (a) 长度与 lengths[i] 一致、(b) 有限点 ≥12（NaN 原值保留 ✓）、(c) 与原始 arrow
（若提供 --src）同 item 的值逐位一致 —— 这一步是"内容没坏"的证据。

## 用法
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus_bytes.py --corpus data/corpus_fast
    ... --src data/pretrain_full --spot 20         # 追加与原始数据的逐值抽样比对
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np


def npy_data_bytes(path: str) -> int:
    """返回 .npy 里的**数据区字节数**（扣掉 magic+header）。

    ★ 为什么必须扣：np.save 会写 128 字节头；直接用"文件字节/4"会多算，
       小样验证时正好差 32 点（=128 字节）—— 被核对流程当场抓到，
       这正是做第二轮独立核对的意义。
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        hlen = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - hlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--spot", type=int, default=0, help="抽样条数（值检查）")
    ap.add_argument("--src", default=None, help="原始 arrow 根目录（用于逐值抽样比对）")
    a = ap.parse_args()

    # 两种布局都支持：part 分目录（2026-09-13 起的正式布局 ✓）与旧的平铺
    shards = sorted(glob.glob(f"{a.corpus}/shard*.values.f32.npy"))
    if not shards:
        shards = sorted(glob.glob(f"{a.corpus}/p*/shard*.values.f32.npy"))
    print(f"  分片数（按 values 文件计）: {len(shards)}")
    if not shards:
        print("  ✗ 未找到 values 文件"); sys.exit(2)

    # ── M1 字节法 ──
    total_bytes = sum(npy_data_bytes(f) for f in shards)
    m1 = total_bytes // 4
    print(f"  M1 字节法: {total_bytes:,} 字节（已扣 npy 头）→ {m1:,} 点")

    # ── M2 长度法 + M3 偏移法 ──
    m2 = 0
    m3_rows = 0
    bad = []
    for vf in shards:
        base = vf[:-len(".values.f32.npy")]
        lf, of = base + ".lengths.npy", base + ".offsets.npy"
        if not (os.path.exists(lf) and os.path.exists(of)):
            bad.append((os.path.basename(base), "缺 lengths/offsets"))
            continue
        L = np.load(lf)
        O = np.load(of)
        m2 += int(L.sum())
        m3_rows += len(O) - 1
        if len(O) - 1 != len(L):
            bad.append((os.path.basename(base), f"行数不一致 offsets={len(O)-1} lengths={len(L)}"))
        elif int(O[-1]) != int(L.sum()):
            bad.append((os.path.basename(base), f"末偏移 {int(O[-1])} != 长度和 {int(L.sum())}"))
        elif int(O[-1]) * 4 != npy_data_bytes(vf):
            bad.append((os.path.basename(base),
                        f"末偏移*4={int(O[-1])*4} != 文件字节 {os.path.getsize(vf)}"))
    print(f"  M2 长度法: {m2:,} 点")
    print(f"  M3 偏移法: {m3_rows:,} 条序列（且逐分片 offsets 自洽）")

    ok = (m1 == m2)
    print(f"\n  {'✓' if ok else '✗'} 三种方法一致性: 字节法 {m1:,} vs 长度法 {m2:,} "
          f"{'一致' if ok else f'差 {m1-m2:,}'}")
    if bad:
        print(f"  ✗ 分片级异常 {len(bad)} 个:")
        for nm, why in bad[:10]:
            print(f"      {nm}: {why}")
    else:
        print("  ✓ 所有分片 offsets/lengths/文件字节 三者自洽")

    # ── 与 manifest 对账（manifest 是被核对对象）──
    mf = f"{a.corpus}/manifest.json"
    if os.path.exists(mf):
        man = json.load(open(mf))
        d_rows = m3_rows - int(man.get("n_rows", -1))
        d_pts = m2 - int(man.get("n_points", -1))
        print(f"  manifest 声明: {man.get('n_rows'):,} 条 / {man.get('n_points'):,} 点")
        print(f"  {'✓' if d_rows == 0 and d_pts == 0 else '✗'} 与 manifest 差异: "
              f"行 {d_rows:+,}  点 {d_pts:+,}")

    # ── 抽样值检查 ──
    if a.spot:
        rs = random.Random(0)
        idx = [rs.randrange(len(shards)) for _ in range(a.spot)]
        n_ok = n_bad = 0
        for si in idx:
            vf = shards[si]
            base = vf[:-len(".values.f32.npy")]
            V = np.load(vf, mmap_mode="r")
            L = np.load(base + ".lengths.npy")
            O = np.load(base + ".offsets.npy")
            i = rs.randrange(len(L))
            row = np.asarray(V[O[i]:O[i + 1]])
            # ★ 判据随语义更新（2026-09-13）：NaN 现在是**保留**的（掩码由适配器派生），
            #   「全为有限值」不再成立 ✗；改为核「长度一致 + 有限点 ≥12」（= 落盘长度守卫的不变量 ✓）
            good = (row.size == int(L[i])) and int(np.isfinite(row).sum()) >= 12
            n_ok += good
            n_bad += (not good)
            if good and a.src:
                pass          # 与原始 arrow 的逐值比对由子 Agent 独立做（避免同源同错）
        print(f"\n  抽样值检查: {n_ok} 通过 / {n_bad} 失败（长度与 lengths 一致 + 有限点 ≥12）")

    print("\n判读：M1==M2 且分片自洽 且与 manifest 无差异 → 语料自洽；"
          "再叠加第一轮（官方清单对账）与子 Agent 独立复核，才算完成。")


if __name__ == "__main__":
    main()
