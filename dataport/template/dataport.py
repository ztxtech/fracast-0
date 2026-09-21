"""新数据端口骨架：一个数据源 = 一个文件 = 一个函数 ✓。

用法：复制本文件 → 改名成 `dataport/<数据源>.py` → 填上真正的读取逻辑，
再在 `dataport/dataport.py::build_train_loaders()` 里按配置键接上 ✓。

约定（与真实实现保持一致 ✓）：
- **函数式**，不做基类/继承抽象 ✗（真实实现 `dataport/build_corpus.py` 是函数，
  `dataport/corpus_dataset.py` 只是继承 `ShardDataset` 的叶子类，不是为了抽象而抽象 ✓）；
- **本文件没有 CLI** ✗，主入口只有根目录 `main.py` ✓；
- 路径与数值一律从 `cfg` 取（配置写在 `config/`；临时配置可以让 `script/make_config.py` 写进 `tmp/` ✓）。
"""
from __future__ import annotations

from pathlib import Path


def read_dataset(root: Path, cfg: dict):
    """把一个数据源的原始文件读成训练/评测直接能用的形式。

    返回什么由数据源自己定：数据集对象、样本列表、或 `(train_loader, val_loader)` 都行 ✓ ——
    只要调用方（`dataport/dataport.py` → `pipeline/train.py`）拿到的语义与既有路径一致 ✓。
    """
    raise NotImplementedError("按具体数据源补上读取逻辑")


def build_loaders(cfg: dict):
    """从 cfg 取根目录与参数，装配 DataLoader（不需要这一步就删掉本函数 ✓）。"""
    root = Path(cfg["data"]["dataset_dir"])
    return read_dataset(root, cfg)
