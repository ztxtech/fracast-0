from __future__ import annotations


from pathlib import Path
import json
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回对应的 Path。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """将数据以 UTF-8 JSON 写入文件，自动创建父目录。"""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return path


def load_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件并返回解析结果。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
