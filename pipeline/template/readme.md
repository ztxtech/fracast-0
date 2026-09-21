# template/ — 流程骨架

## 目录用途

`pipeline/` 下新流程的起手骨架：把 `pipeline.py` 复制成 `pipeline/<新流程>.py`，再实现 `run(config)`
（或直接按四类流程之一扩写 `pipeline/<名字>.py` ✓）。

## 包含什么

- `pipeline.py`：流程骨架 —— 只有 `run(config)` 一个口，**没有** `main()` / `argparse` ✓
  （分派那个 `Pipeline` 类是 `pipeline/pipeline.py`，不要与本骨架混为一谈 ✓）

## 不包含什么

- 具体流程实现（四类流程在 `pipeline/` 顶层：`build_corpus.py` / `eval_ref.py` / `train.py` / `eval_ours.py`）✗
- 通用工具（放 `util/`）✗

## 命名与组织约定

- 新流程一个文件一个流程，并在 `pipeline/pipeline.py` 的 `kind` 分派里加一行 ✓
- 模板保持最小可运行、易复制；参数与流程细节不写进模板 ✓
- 流程**不得**在 `pipeline/` 里写模型定义/数据读取 ✗ → `model/`、`dataport/` ✓
