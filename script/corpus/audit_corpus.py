"""语料完整性对账（方法 A：按 HF 官方文件清单 vs 本地实际落盘，逐数据集比对）。

## 为什么需要
用户要求："所有公开数据都要获取到……转换完之后要反复核对反复核对"。
文件数相等**不等于**数据完整（可能文件在但内容截断/坏），所以本脚本只做**第一轮**：
**按数据集名逐一**比对 官方清单（HF API） vs 本地落盘（文件数 + 字节数），
输出缺口清单。第二轮（逐字节/逐序列）由 `audit_corpus_bytes.py` 承担，
第三轮由**无上下文子 Agent** 独立复核。

## 用法
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus.py --repo Salesforce/GiftEvalPretrain --dir data/pretrain_full
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus.py --all      # 三个仓库一起
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

# 数据一律在项目内 data/ ✓
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
REPOS = {
    "pret": ("Salesforce/GiftEvalPretrain", os.path.join(DATA, "pretrain_full")),
    "lotsa": ("Salesforce/lotsa_data", os.path.join(DATA, "lotsa_full")),
    "chronos": ("autogluon/chronos_datasets", os.path.join(DATA, "chronos_full")),
}


def local_stats(d: str):
    """本地：{数据集: (文件数, 字节数)}（首层目录为数据集名）"""
    out = defaultdict(lambda: [0, 0])
    if not os.path.isdir(d):
        return out
    for root, _dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        if rel == ".":
            top = "(root)"
        else:
            top = rel.split(os.sep)[0]
        for f in files:
            if f.endswith((".lock", ".incomplete")) or f.startswith("."):
                continue
            p = os.path.join(root, f)
            try:
                out[top][0] += 1
                out[top][1] += os.path.getsize(p)
            except OSError:
                pass
    return out


def official_stats(repo: str):
    """官方：{数据集: (文件数, 字节数)}"""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo, files_metadata=True)
    out = defaultdict(lambda: [0, 0])
    for f in info.siblings:
        name = f.rfilename
        if name.startswith("."):
            continue
        top = name.split("/")[0] if "/" in name else "(root)"
        out[top][0] += 1
        out[top][1] += f.size or 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    a = ap.parse_args()

    targets = []
    if a.all:
        targets = [(k, r, d) for k, (r, d) in REPOS.items()]
    elif a.repo and a.dir:
        targets = [("custom", a.repo, a.dir)]
    else:
        ap.error("需要 --repo/--dir 或 --all")

    report = {}
    for tag, repo, d in targets:
        off = official_stats(repo)
        loc = local_stats(d)
        n_off_files = sum(v[0] for v in off.values())
        n_loc_files = sum(v[0] for v in loc.values())
        b_off = sum(v[1] for v in off.values())
        b_loc = sum(v[1] for v in loc.values())
        print(f"\n═══ {tag}: {repo}")
        print(f"  官方: {len(off):3d} 数据集 / {n_off_files:5d} 文件 / {b_off/1e9:7.1f} GB")
        print(f"  本地: {len(loc):3d} 数据集 / {n_loc_files:5d} 文件 / {b_loc/1e9:7.1f} GB"
              f"  完成度 {100*b_loc/max(b_off,1):5.1f}%（按字节）")
        # 逐数据集差异（按字节）
        diff = []
        for ds, (nf, nb) in sorted(off.items()):
            lf, lb = loc.get(ds, [0, 0])
            if lb < nb * 0.999:                     # 允许 0.1% 的文件系统开销
                diff.append((ds, nb, lb, nf, lf))
        diff.sort(key=lambda x: -(x[1] - x[2]))
        print(f"  未完整的数据集: {len(diff)}/{len(off)}")
        for ds, nb, lb, nf, lf in diff[:12]:
            print(f"    {ds:42s} {lb/1e9:7.2f}/{nb/1e9:7.2f} GB  ({lf}/{nf} 文件)"
                  f"  缺 {(nb-lb)/1e9:6.2f} GB")
        if len(diff) > 12:
            print(f"    … 另有 {len(diff)-12} 个数据集未完整")
        report[tag] = {
            "repo": repo, "dir": d,
            "official": {"datasets": len(off), "files": n_off_files, "bytes": b_off},
            "local": {"datasets": len(loc), "files": n_loc_files, "bytes": b_loc},
            "incomplete": [{"dataset": ds, "official_bytes": nb, "local_bytes": lb,
                            "official_files": nf, "local_files": lf} for ds, nb, lb, nf, lf in diff],
        }
        missing_ds = sorted(set(off) - set(loc))
        if missing_ds:
            print(f"  完全缺失的数据集（{len(missing_ds)}）: {missing_ds[:10]}")

    if a.json:
        json.dump(report, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n✓ 报告写入 {a.json}")
    print("\n判读：本地字节 ≥ 官方 99.9% 才算该数据集完整；"
          "全部完整后进行第二轮（逐字节/逐序列）与子 Agent 复核。")


if __name__ == "__main__":
    main()
