"""Repository-local path constants.

Every runtime artifact stays inside the checkout by default:

* ``data/``: downloaded or generated datasets (gitignored)
* ``output/``: training runs and checkpoints (gitignored)
* ``tmp/``: smoke tests, logs, and disposable intermediates (gitignored)
* ``.cache/``: Hugging Face downloads unless ``HF_HOME`` is set
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
TMP: Path = ROOT / "tmp"
OUTPUT: Path = ROOT / "output"
DATA: Path = ROOT / "data"
HF_CACHE: Path = Path(os.environ.get("HF_HOME", str(ROOT / ".cache" / "huggingface")))

__all__ = ["ROOT", "TMP", "OUTPUT", "DATA", "HF_CACHE"]
