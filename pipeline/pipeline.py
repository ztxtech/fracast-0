"""Config-driven pipeline dispatch for corpus preparation and pretraining.

调用链（唯一一条 ✓）：

    main.py <配置|目录>            ← 唯一 CLI：读配置、展开网格、批量调度
      └─ Pipeline(config).run()    ← 本文件：按 `_run.kind` 跳到下面的方法
           └─ pipeline/<流程>.py::run(config)   ← 流程实现（**没有 main / 没有 argparse** ✗）
                └─ model/ · module/ · dataport/ ← 模型、组件、数据实现

参数只从**配置**来 ✓（临时任务就在 `tmp/` 里写一份临时配置再交给 `main.py` ✓）；
要改某个流程，只改对应文件里的 `run(config)`，入口与调度都不用动 ✓。

**pipeline 下只有两类流程**：

| kind | 流程 | 实现 |
|---|---|---|
| `build_corpus` | 原始 arrow/parquet 语料 → 最快读取格式 | `pipeline/build_corpus.py` → `dataport/build_corpus.py` |
| `train` | pretrain FracCast | `pipeline/train.py` |

不在本层的内容：
  · 模型定义 → `model/` ✓
  · 数据读取与格式转换实现 → `dataport/` ✓
  · GPU 池/占卡等运维调度 → `script/pool/` ✓
  · 诊断工具 → `script/diagnostics/` ✓
"""
from __future__ import annotations

from typing import Any

# 两类流程（kind ✓）—— main.py 按这份清单校验配置，写错 kind 早失败 ✓
KINDS = ("build_corpus", "train")
_ALIASES = {"train_eval": "train"}


class Pipeline:
    """一次实验的端到端流程：吃 config，返回结果 ✓。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    @property
    def kind(self) -> str:
        """流程类型：`_run.kind`（我们的配置约定 ✓）优先，其次 `pipeline.name`。"""
        run_meta = self.config.get("_run") or {}
        raw = str(run_meta.get("kind")
                  or (self.config.get("pipeline") or {}).get("name")
                  or "train")
        return _ALIASES.get(raw, raw)

    def flow_config(self) -> dict[str, Any]:
        """给流程实现的纯配置：剥掉 `_run` 等元数据（不会写进 `config_used.yaml` ✓）。"""
        from util.config import strip_meta

        return strip_meta(self.config)

    # ── supported flows ─────────────────────────────────────────────────────
    def build_corpus(self) -> Any:
        """原始语料 → 快速读取格式（`pipeline/build_corpus.py` ✓）。"""
        from pipeline.build_corpus import run
        return run(self.flow_config())

    def train(self) -> Any:
        """训练（`pipeline/train.py` ✓）。"""
        from pipeline.train import run
        return run(self.flow_config())

    def run(self) -> Any:
        method = {"build_corpus": self.build_corpus,
                  "train": self.train}.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"Pipeline 未实现的 kind: {self.kind!r}；应为 {' / '.join(KINDS)}")
        return method()
