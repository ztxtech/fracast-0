"""Check that a temporary Fracast TIME result directory matches the official layout."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_FILES = ("config.json", "metrics.npz", "predictions.npz")
REQUIRED_CONFIG = {
    "num_series", "num_windows", "num_variates", "prediction_length",
    "num_quantiles", "quantile_levels", "freq", "seasonality", "context_length",
    "model", "rollout_chunks", "weights",
}


def validate(model_dir: Path, expected_configs: set[str]) -> list[Path]:
    configs = {
        str(path.relative_to(model_dir))
        for path in model_dir.glob("*/*/*")
        if path.is_dir()
    }
    missing = sorted(expected_configs - configs)
    if missing:
        raise RuntimeError(f"missing TIME configurations: {missing}")
    unexpected = sorted(configs - expected_configs)
    if unexpected:
        raise RuntimeError(f"unexpected TIME configurations: {unexpected}")

    for directory in sorted(model_dir.glob("*/*/*")):
        for filename in EXPECTED_FILES:
            path = directory / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing or empty {path}")
        with (directory / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("model") != "Fracast-0":
            raise RuntimeError(f"unexpected model in {directory}: {config.get('model')}")
        if not REQUIRED_CONFIG <= config.keys():
            raise RuntimeError(f"missing provenance keys in {directory / 'config.json'}")
        with np.load(directory / "metrics.npz") as metrics:
            for metric_name in ("MASE", "CRPS", "MAE", "MSE"):
                if metric_name not in metrics.files:
                    raise RuntimeError(f"missing {metric_name} in {directory / 'metrics.npz'}")
                metric_values = metrics[metric_name]
                if not np.isfinite(metric_values).any():
                    raise RuntimeError(f"all-{metric_name} non-finite in {directory / 'metrics.npz'}")
                if not math.isfinite(float(np.nanmean(metric_values))):
                    raise RuntimeError(f"non-finite mean {metric_name} in {directory / 'metrics.npz'}")
        with np.load(directory / "predictions.npz") as predictions:
            if set(predictions.files) != {"predictions_quantiles", "quantile_levels"}:
                raise RuntimeError(f"unexpected prediction keys in {directory}")
    return sorted(path for path in model_dir.glob("*/*/*") if path.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expected", nargs="+", required=True, help="dataset/term paths")
    args = parser.parse_args()
    directories = validate(args.model_dir, set(args.expected))
    print(f"PASS: {len(directories)} official TIME task directories")
    for directory in directories:
        print(directory)


if __name__ == "__main__":
    main()
