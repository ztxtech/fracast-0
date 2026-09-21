from __future__ import annotations


from pathlib import Path
import json
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Write UTF-8 JSON and create parent directories."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return path


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
