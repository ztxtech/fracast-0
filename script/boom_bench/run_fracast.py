#!/usr/bin/env python3
"""Evaluate a Fracast release with the official BOOM protocol."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
BOOM_REVISION = "69325b544c45ff0d6c43c7a99c49a6601a01725b"
GIFT_EVAL_REVISION = "1527c41589189ad1bc3883ed4d3d97b3e5a3b47c"
METRIC_COLUMNS = [
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
]
CSV_COLUMNS = ["dataset", "model", *METRIC_COLUMNS, "domain", "num_variates", "dataset_size"]
TERMS = ("short", "medium", "long")


@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    term: str

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.term}"


def load_properties(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_configs(properties: dict[str, dict], benchmark: str) -> Iterator[EvalConfig]:
    names = sorted(properties)
    for name in names:
        for term in TERMS:
            if term != "short" and properties[name]["term"] == "short":
                continue
            yield EvalConfig(name, term)


def completed_configs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"] for row in csv.DictReader(handle)}


def write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(CSV_COLUMNS)


def jsonable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def append_result(path: Path, values: list) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(values)
        handle.flush()
        os.fsync(handle.fileno())


def download_data(
    benchmark: str,
    data_root: Path,
    properties_path: Path,
    token: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    properties = load_properties(properties_path)
    patterns = ["README.md", "dataset_taxonomy.json"]
    if benchmark == "boomlet":
        patterns.extend(f"{name}/*" for name in sorted(properties))
    else:
        patterns.append("ds-*/*")
    snapshot_download(
        repo_id="Datadog/BOOM",
        repo_type="dataset",
        revision=BOOM_REVISION,
        local_dir=data_root,
        allow_patterns=patterns,
        token=token,
    )
    return data_root


class FracastPredictor:
    """Adapt batched Fracast forecasts to the official GluonTS interface."""

    def __init__(self, model, prediction_length: int, batch_size: int):
        self.model = model
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)

    def predict(self, test_data_input: Iterable[dict], batch_size: int | None = None):
        from gluonts.model.forecast import QuantileForecast

        width = self.batch_size if batch_size is None else int(batch_size)
        batch: list[dict] = []
        for item in test_data_input:
            batch.append(item)
            if len(batch) < width:
                continue
            yield from self._forecast_batch(batch)
            batch = []
        if batch:
            yield from self._forecast_batch(batch)

    def _forecast_batch(self, batch: list[dict]):
        from gluonts.model.forecast import QuantileForecast

        context = [np.asarray(item["target"], dtype=np.float32) for item in batch]
        max_length = max(len(values) for values in context)
        padded = np.full(
            (len(context), max_length), np.nan, dtype=np.float32
        )
        for row, values in enumerate(context):
            padded[row, max_length - len(values):] = values
        forecasts = self.model.forecast_batch(
            padded, horizon=self.prediction_length
        )
        quantiles = [str(value) for value in self.model.quantiles]
        for item, forecast in zip(batch, forecasts):
            yield QuantileForecast(
                forecast_arrays=np.asarray(forecast, dtype=np.float32).T,
                forecast_keys=quantiles,
                start_date=item["start"] + len(item["target"]),
                item_id=item["item_id"],
            )


def _metric_value(result: dict, key: str):
    value = result[key.removeprefix("eval_metrics/")]
    return value.iloc[0] if hasattr(value, "iloc") else value[0]


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    properties_file = (
        "boomlet_properties.json" if args.benchmark == "boomlet" else "boom_properties.json"
    )
    properties_path = root / "script" / "boom_bench" / properties_file
    properties = load_properties(properties_path)
    configs = list(iter_configs(properties, args.benchmark))
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        configs = [config for config in configs if config.name in wanted or config.dataset in wanted]
    if args.limit > 0:
        keep_datasets: list[str] = []
        limited: list[EvalConfig] = []
        for config in configs:
            if config.dataset not in keep_datasets:
                if len(keep_datasets) >= args.limit:
                    continue
                keep_datasets.append(config.dataset)
            limited.append(config)
        configs = limited

    output_csv = args.output_dir / "all_results.csv"
    manifest_path = args.output_dir / "run-manifest.json"
    done = completed_configs(output_csv)
    pending = [config for config in configs if config.name not in done]
    write_manifest(
        manifest_path,
        {
            "arguments": jsonable_args(args),
            "benchmark": args.benchmark,
            "boom_revision": BOOM_REVISION,
            "gift_eval_revision": GIFT_EVAL_REVISION,
            "completed": len(done),
            "pending": len(pending),
            "total_configs": len(configs),
        },
    )
    if not pending:
        print(f"[boom] {len(done)}/{len(configs)} configurations already complete")
        return 0

    if args.download:
        download_data(args.benchmark, args.data_root, properties_path, args.hf_token)
    if not args.data_root.exists():
        raise FileNotFoundError(
            f"BOOM data root does not exist: {args.data_root}; use --download"
        )

    os.environ["BOOM"] = str(args.data_root.resolve())
    from gift_eval.data import Dataset
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    from fracast import FracastModel

    class _ForecastLogFilter(logging.Filter):
        def filter(self, record):
            return (
                "The mean prediction is not stored in the forecast data"
                not in record.getMessage()
            )

    logging.getLogger("gluonts.model.forecast").addFilter(_ForecastLogFilter())

    write_header(output_csv)
    model = FracastModel.from_pretrained(
        args.model, weights=args.weights, device=args.device
    )
    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=model.quantiles),
    ]
    failures: list[str] = []
    started = time.monotonic()
    for index, config in enumerate(pending, start=1):
        print(
            f"[boom] {index}/{len(pending)} {config.name} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        try:
            probe = Dataset(
                config.dataset,
                term=config.term,
                to_univariate=False,
                storage_env_var="BOOM",
            )
            to_univariate = probe.target_dim != 1
            dataset = Dataset(
                config.dataset,
                term=config.term,
                to_univariate=to_univariate,
                storage_env_var="BOOM",
            )
            dataset_size = len(dataset.test_data)
            predictor = FracastPredictor(
                model,
                prediction_length=dataset.prediction_length,
                batch_size=args.batch_size,
            )
            result = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=args.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(dataset.freq),
            )
            append_result(
                output_csv,
                [
                    config.name,
                    args.model_name,
                    *(_metric_value(result, column) for column in METRIC_COLUMNS),
                    json.dumps(properties[config.dataset]["domain"]),
                    properties[config.dataset]["num_variates"],
                    dataset_size,
                ],
            )
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append(f"{config.name}: {exc!r}")
            print(f"[boom] FAILED {config.name}: {exc!r}", file=sys.stderr, flush=True)
        write_manifest(
            manifest_path,
            {
                "arguments": jsonable_args(args),
                "benchmark": args.benchmark,
                "boom_revision": BOOM_REVISION,
                "gift_eval_revision": GIFT_EVAL_REVISION,
                "completed": len(completed_configs(output_csv)),
                "pending": len(configs) - len(completed_configs(output_csv)),
                "total_configs": len(configs),
                "failures": failures,
            },
        )

    final_done = len(completed_configs(output_csv))
    print(f"[boom] complete {final_done}/{len(configs)} in {time.monotonic() - started:.1f}s")
    return 0 if not failures else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("boom", "boomlet"), default="boomlet")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "boom")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "boom_bench" / "fracast-0")
    parser.add_argument("--model", default=None, help="Defaults to bundled Fracast-0 weights")
    parser.add_argument("--weights", choices=("fp32", "w8"), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", default="fracast-0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Smoke limit by metric query")
    parser.add_argument("--only", help="Comma-separated dataset or dataset/term names")
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
