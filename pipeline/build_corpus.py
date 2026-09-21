"""Convert raw Arrow/Parquet data into the repository's fast corpus format.

This module maps configuration values to the format implementation in
``dataport/build_corpus.py``. It intentionally has no command-line interface;
the supported entry point is ``main.py`` with a YAML configuration:

    python main.py config/corpus/<name>.yaml

Supported keys include ``src``, ``out``, ``shard_gb``, ``parts``, ``part``,
``balance``, ``max_files``, ``dry``, ``file_mod``, and ``file_mod_n``.
Parallel conversion is expressed as a configuration grid over ``part``.
"""
from __future__ import annotations


def run(config: dict) -> dict:
    """Convert a source dataset and return metrics for the experiment record."""
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
