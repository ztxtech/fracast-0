# template — 数据端口模板

## 目录用途

存放新数据端口的起手骨架：复制一份改个名就是一个新数据源 ✓。

## 包含什么

- `dataport.py`：函数式骨架（`read_dataset(root, cfg)` + 可选 `build_loaders(cfg)`），
  不做基类/继承抽象 ✗，与真实实现（`dataport/dataport.py` 分派 + 各数据源文件）一致 ✓。

## 不包含什么

- 初始数据文件（放 `data/`）。
- 具体数据端口实现（放 `dataport/` 下）。

## 命名与组织约定

- 一个数据源一个文件；新增后同步更新 `dataport/readme.md` 与 `dataport/dataport.py` 的分派 ✓。
- 模板保持最小可读，不留 `if __name__ == "__main__"` / `argparse` ✗（主入口只有 `main.py` ✓）。
