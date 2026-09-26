"""Run the official GIFT-Eval 97-configuration protocol for Fracast-0."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fracast import FracastModel  # noqa: E402


logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)


MODEL_NAME = "Fracast-0"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HEADER = [
    "dataset",
    "model",
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
    "domain",
    "num_variates",
]

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T "
    "ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T "
    "jena_weather/H jena_weather/D bitbrains_fast_storage/5T "
    "bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()

MEDIUM_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application "
    "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
    "loop_seattle": "loop_seattle",
    "m_dense": "m_dense",
    "sz_taxi": "sz_taxi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Fracast-0 with the official GIFT-Eval protocol."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GIFT_EVAL", "")),
        help="GiftEval directory containing the 28 dataset roots. "
        "Defaults to GIFT_EVAL when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / MODEL_NAME,
        help="Submission directory. Defaults to results/Fracast-0.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=ROOT / "weights",
        help="Release checkpoint directory.",
    )
    parser.add_argument(
        "--weights",
        choices=("fp32", "w8"),
        default="fp32",
        help="Checkpoint format. Official submissions use fp32.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of test windows predicted in one model call.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CONFIG",
        help="Run only the listed dataset configurations for a smoke test.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def canonicalize_dataset(disk_name: str) -> tuple[str, str]:
    if "/" in disk_name:
        dataset, frequency = disk_name.split("/", 1)
    else:
        dataset = disk_name
        frequency = ""
    dataset = PRETTY_NAMES.get(dataset.lower(), dataset.lower())
    return dataset, frequency


def load_properties() -> dict:
    path = ROOT / "config" / "gift_eval" / "dataset_properties.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_configurations() -> list[dict]:
    properties = load_properties()
    disk_names = sorted(set(SHORT_DATASETS + MEDIUM_LONG_DATASETS))
    configurations: list[dict] = []
    for disk_name in disk_names:
        terms = ["short"]
        if disk_name in MEDIUM_LONG_DATASETS:
            terms.extend(["medium", "long"])

        dataset_name, disk_frequency = canonicalize_dataset(disk_name)
        frequency = disk_frequency or properties[dataset_name]["frequency"]
        metadata = properties[dataset_name]
        for term in terms:
            configurations.append(
                {
                    "disk_name": disk_name,
                    "dataset_name": dataset_name,
                    "frequency": frequency,
                    "term": term,
                    "id": f"{dataset_name}/{frequency}/{term}",
                    "domain": metadata["domain"],
                    "num_variates": metadata["num_variates"],
                }
            )

    configurations.sort(key=lambda item: item["id"])
    identifiers = [item["id"] for item in configurations]
    if len(configurations) != 97 or len(set(identifiers)) != 97:
        raise RuntimeError(
            f"expected 97 unique configurations, got {len(configurations)}"
        )
    return configurations


def metric_objects() -> list:
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

    return [
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
        MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
    ]


class FracastPredictor:
    """Batched GluonTS predictor for the released channel-independent model."""

    def __init__(self, model: FracastModel, prediction_length: int, batch_size: int):
        self.model = model
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)

    @staticmethod
    def _batch_context(entries: Sequence[dict]) -> np.ndarray:
        arrays = [np.asarray(entry["target"], dtype=np.float32) for entry in entries]
        if any(array.ndim != 1 for array in arrays):
            raise ValueError(
                "the evaluator expected univariate test inputs; got "
                f"{[array.shape for array in arrays]}"
            )
        max_length = max(array.shape[0] for array in arrays)
        context = np.full((len(arrays), max_length), np.nan, dtype=np.float32)
        for row, array in enumerate(arrays):
            context[row, -array.shape[0] :] = array
            tail_start = max(0, context.shape[1] - 2048)
            tail = context[row, tail_start:]
            finite = np.isfinite(tail)
            if int(finite.sum()) < 8:
                # A few official windows end in a missing-value run. Use the
                # nearest observed value as the local baseline so the release
                # model can apply its normal masked forward pass.
                full_finite = np.flatnonzero(np.isfinite(context[row]))
                fill_value = (
                    context[row, full_finite[-1]] if full_finite.size else np.float32(0)
                )
                tail[~finite] = fill_value
        return context

    def predict(
        self, test_data_input: Iterable[dict], batch_size: int | None = None
    ) -> list:
        from gluonts.itertools import batcher
        from gluonts.model.forecast import QuantileForecast

        size = self.batch_size if batch_size is None else int(batch_size)
        if size < 1:
            raise ValueError(f"batch size must be positive, got {size}")

        forecasts = []
        for batch in batcher(test_data_input, batch_size=size):
            arrays = self._batch_context(batch)
            predictions = self.model.forecast_batch(
                arrays, horizon=self.prediction_length
            )
            if predictions.shape != (
                len(batch),
                self.prediction_length,
                len(QUANTILE_LEVELS),
            ):
                raise RuntimeError(
                    "unexpected forecast shape: "
                    f"{predictions.shape} for batch of {len(batch)}"
                )
            for entry, prediction in zip(batch, predictions):
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=prediction.T,
                        forecast_keys=[str(level) for level in QUANTILE_LEVELS],
                        start_date=entry["start"] + len(entry["target"]),
                    )
                )
        return forecasts


def completed_configurations(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != HEADER:
            raise RuntimeError(f"unexpected resume header in {path}: {header}")
        return {row[0] for row in reader if row}


def ensure_results_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(HEADER)


def extract_row(result, configuration: dict) -> list:
    row = [
        configuration["id"],
        MODEL_NAME,
        result["MSE[mean]"].iloc[0],
        result["MSE[0.5]"].iloc[0],
        result["MAE[0.5]"].iloc[0],
        result["MASE[0.5]"].iloc[0],
        result["MAPE[0.5]"].iloc[0],
        result["sMAPE[0.5]"].iloc[0],
        result["MSIS"].iloc[0],
        result["RMSE[mean]"].iloc[0],
        result["NRMSE[mean]"].iloc[0],
        result["ND[0.5]"].iloc[0],
        result["mean_weighted_sum_quantile_loss"].iloc[0],
        configuration["domain"],
        configuration["num_variates"],
    ]
    numeric = np.asarray(row[2:13], dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError(
            f"non-finite metric for {configuration['id']}: {row[2:13]}"
        )
    return row


def validate_submission(csv_path: Path, config_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = [row["dataset"] for row in rows]
    expected = {item["id"] for item in build_configurations()}
    if len(rows) != 97 or len(set(identifiers)) != 97:
        raise RuntimeError(
            f"expected 97 unique result rows, got {len(rows)} "
            f"({len(set(identifiers))} unique)"
        )
    if set(identifiers) != expected:
        missing = sorted(expected - set(identifiers))
        extra = sorted(set(identifiers) - expected)
        raise RuntimeError(f"configuration mismatch; missing={missing}, extra={extra}")
    for row in rows:
        values = [float(row[HEADER[index]]) for index in range(2, 13)]
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite metric in {row['dataset']}")

    submission = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "model",
        "model_type",
        "model_dtype",
        "model_link",
        "code_link",
        "org",
        "testdata_leakage",
        "replication_code_available",
    }
    if set(submission) != required or submission["model"] != MODEL_NAME:
        raise RuntimeError(f"unexpected submission config: {submission}")
    print(f"validated {len(rows)} rows, {len(HEADER)} columns, and config.json")


def write_submission_config(path: Path) -> None:
    payload = {
        "model": MODEL_NAME,
        "model_type": "pretrained",
        "model_dtype": "float32",
        "model_link": "https://huggingface.co/ztxtech/fracast-0",
        "code_link": "https://github.com/ztxtech/fracast-0/blob/main/script/gift_eval/evaluate.py",
        "org": "ztxtech",
        "testdata_leakage": "Yes",
        "replication_code_available": "Yes",
    }
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not str(args.data_root):
        raise SystemExit("--data-root or GIFT_EVAL is required")
    if not args.data_root.is_dir():
        raise SystemExit(f"GiftEval data root does not exist: {args.data_root}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    configurations = build_configurations()
    if args.only:
        selected = set(args.only)
        unknown = selected - {item["id"] for item in configurations}
        if unknown:
            raise SystemExit(f"unknown configurations: {sorted(unknown)}")
        configurations = [
            item for item in configurations if item["id"] in selected
        ]

    os.environ["GIFT_EVAL"] = str(args.data_root.resolve())
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    device = resolve_device(args.device)
    model = FracastModel.from_pretrained(
        args.weights_dir, weights=args.weights, device=device
    )
    csv_path = args.output_dir / "all_results.csv"
    config_path = args.output_dir / "config.json"
    ensure_results_header(csv_path)
    completed = completed_configurations(csv_path)

    print(
        f"device={device} weights={args.weights} batch_size={args.batch_size} "
        f"resume={len(completed)}/97",
        flush=True,
    )
    for index, configuration in enumerate(configurations, start=1):
        identifier = configuration["id"]
        if identifier in completed:
            print(f"[{index}/{len(configurations)}] skip {identifier}", flush=True)
            continue

        probe = Dataset(
            name=configuration["disk_name"],
            term=configuration["term"],
            to_univariate=False,
        )
        to_univariate = probe.target_dim > 1
        dataset = Dataset(
            name=configuration["disk_name"],
            term=configuration["term"],
            to_univariate=to_univariate,
        )
        predictor = FracastPredictor(
            model=model,
            prediction_length=dataset.prediction_length,
            batch_size=args.batch_size,
        )
        print(
            f"[{index}/{len(configurations)}] {identifier} "
            f"windows={len(dataset.test_data)} "
            f"horizon={dataset.prediction_length} "
            f"expanded={to_univariate}",
            flush=True,
        )
        result = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metric_objects(),
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(dataset.freq),
        )
        row = extract_row(result, configuration)
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        completed.add(identifier)
        print(
            f"  MASE={row[5]:.6f} MWQL={row[12]:.6f} "
            f"sMAPE={row[7]:.6f}",
            flush=True,
        )

    if args.only:
        print("smoke evaluation finished without writing config.json", flush=True)
        return 0

    write_submission_config(config_path)
    validate_submission(csv_path, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
