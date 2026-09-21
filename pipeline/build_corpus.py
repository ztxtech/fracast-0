"""流程：原始 arrow/parquet 语料 → 最快读取格式（连续 mmap + offsets ✓）。

边界（用户 2026-09-13 定）：本文件只管**流程**（把 config 映射成格式实现的参数）✓；
格式实现（读取器 + 分片写出）在 `dataport/build_corpus.py` ✓。

**本文件没有 CLI** ✗ —— 唯一入口是根目录 `main.py`（配置驱动 ✓）：

    .venv/bin/python main.py config/corpus/<名>.yaml

config 键：`src`（必填）、`out`（必填）、`shard_gb`、`parts`、`part`、`balance`、
`max_files`、`dry`、`file_mod` / `file_mod_n`（数据集内并行：只读第 `file_mod` 份文件）✓。
并行 = 配置里的网格：`_run.grid: {part: [0, 1, … 31]}` + `main.py … --workers 32`
（一个 part 一个子进程，各自写自己的分片 ✓）。
"""
from __future__ import annotations


def run(config: dict) -> dict:
    """按 config 把源语料转成 fast 格式，返回汇总（数值项会进 run 的 metrics ✓）。"""
    from dataport.build_corpus import build_corpus

    return build_corpus(
        src=str(config["src"]),
        out=str(config["out"]),
        shard_gb=float(config.get("shard_gb", 8.0)),
        max_files=int(config.get("max_files") or 0),
        dry=bool(config.get("dry")),
        part=int(config.get("part") or 0),
        parts=int(config.get("parts") or 1),
        balance=str(config.get("balance") or "index"),
        file_mod=int(config.get("file_mod") or 0),
        file_mod_n=int(config.get("file_mod_n") or 1),
    )
