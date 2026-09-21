#!/usr/bin/env python3
"""Configuration-driven entry point for FracCast pretraining.

Examples:
    python main.py config/corpus/demo.yaml
    python main.py config/fraccast/pretrain_smoke.yaml
    python main.py config/corpus --list
    python main.py config/fraccast/pretrain_full.yaml

Only two flows are supported:

* ``build_corpus``: raw Arrow/Parquet files -> fast mmap corpus
* ``train``: pretrain FracCast from a fast corpus

Pipeline-specific parameters are read from YAML. The CLI only controls
discovery, dry runs, overrides, and optional process-level parallelism.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict[str, Any]:
    """Load one YAML config, preserving the ``_run`` metadata block."""
    from util.config import load_config as load_body

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    cfg = load_body(path)
    cfg["_run"] = dict(raw.get("_run") or {})
    return cfg


def discover_configs(target: Path) -> list[Path]:
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"config path does not exist: {target}")
    return sorted(
        p for p in target.rglob("*")
        if p.suffix.lower() in {".yaml", ".yml"}
        and not p.name.startswith(".")
        and "template" not in p.parts
    )


def expand_configs(paths: list[Path], overrides: list[str]) -> list[dict[str, Any]]:
    from util.config import apply_overrides
    from util.grid import expand_grid

    variants: list[dict[str, Any]] = []
    for path in paths:
        cfg = apply_overrides(load_config(path), overrides)
        variants.extend(expand_grid(cfg))
    return variants


def describe(cfg: dict[str, Any]) -> str:
    meta = cfg.get("_run") or {}
    out = (cfg.get("train") or {}).get("out_dir") or cfg.get("out") or "-"
    return (
        f"kind={meta.get('kind', 'train')!s:12s} "
        f"tag={meta.get('tag', '-')!s:24s} out={out}"
    )


def _run_one(cfg: dict[str, Any]) -> dict[str, Any]:
    gpu = (cfg.get("_run") or {}).get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from pipeline import Pipeline

    return Pipeline(cfg).run()


def run_all(variants: list[dict[str, Any]], workers: int) -> None:
    if workers <= 1 or len(variants) <= 1:
        for cfg in variants:
            print(f"[run] {describe(cfg)}", flush=True)
            _run_one(cfg)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, copy.deepcopy(cfg)): cfg for cfg in variants}
        for fut in as_completed(futures):
            cfg = futures[fut]
            result = fut.result()
            print(f"[done] {describe(cfg)} -> {result}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FracCast config-driven corpus and pretraining entry point.",
    )
    parser.add_argument("config", type=Path, help="YAML config file or directory")
    parser.add_argument("--list", action="store_true", help="list expanded runs")
    parser.add_argument("--dry-run", action="store_true", help="validate and print only")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value; repeatable",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel worker count")
    args = parser.parse_args(argv)

    paths = discover_configs(args.config)
    if not paths:
        raise SystemExit(f"no config files found under {args.config}")
    variants = expand_configs(paths, args.override)
    if args.list or args.dry_run:
        for cfg in variants:
            print(describe(cfg))
        print(f"[main] {len(paths)} config file(s) -> {len(variants)} run(s)")
        return
    run_all(variants, max(1, int(args.workers)))


if __name__ == "__main__":
    main()
