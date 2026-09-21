"""Template for a new data source adapter.

Copy this file to ``dataport/<source>.py``, implement the reader, and add the
configuration dispatch in ``dataport/dataport.py``.  Keep the adapter
functional and configuration-driven.
"""
from __future__ import annotations

from pathlib import Path


def read_dataset(root: Path, cfg: dict):
    """Read one source into a dataset object or an iterable of samples."""
    raise NotImplementedError("implement the reader for this data source")


def build_loaders(cfg: dict):
    """Build loaders from ``cfg`` when a loader-level wrapper is needed."""
    root = Path(cfg["data"]["dataset_dir"])
    return read_dataset(root, cfg)
