"""流程骨架：复制本文件成 `pipeline/<新流程>.py`，只实现一个口 `run(config)` ✓。

硬规则（用户 2026-09-13 定）：
  · **主入口只有根目录 `main.py`** ✓ —— 流程文件里**没有** `main()` / `argparse` ✗；
  · 流程参数全从 config 来：写在实验配置文件里（临时任务用 `script/make_config.py` 生成
    `tmp/` 下的临时配置 ✓），**不在命令行里一条条传** ✗；
  · 写完后在 `pipeline/pipeline.py` 的 `KINDS` 与分派表里各加一行（那是唯一清单 ✓）；
  · 流程只管编排：模型 → `model/`，组件 → `module/`，数据实现 → `dataport/` ✗不在本层写。
"""
from __future__ import annotations


def run(config: dict) -> dict:
    """按 config 跑一次流程；返回的数值项会写进该 run 的 `metrics.json` ✓。"""
    raise NotImplementedError("复制本文件到 pipeline/<名字>.py 后实现 run(config)")
