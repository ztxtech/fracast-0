"""TIME benchmark runner for the released Fracast-0 model.

This is evaluation-only glue. It uses pinned local copies of the official TIME
implementation and this repository's Fracast implementation; it is not part of
the Fracast package API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path


QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_NAME = "Fracast-0"
HF_DATASET = "Real-TSF/TIME"


def load_sources(args: argparse.Namespace) -> dict:
    """Import the pinned local TIME and Fracast implementations."""
    if not args.time_source.is_dir() or not args.package_source.is_dir():
        raise FileNotFoundError(
            "both --time-source and --package-source must be local repository directories"
        )
    sys.path.insert(0, str(args.package_source.resolve()))
    sys.path.insert(0, str((args.time_source / "src").resolve()))

    from fracast import FracastModel
    from timebench.evaluation import data as time_data
    from timebench.evaluation import saver as time_saver
    from timebench.evaluation import utils as time_utils

    return {
        "FracastModel": FracastModel,
        "Dataset": time_data.Dataset,
        "get_dataset_settings": time_data.get_dataset_settings,
        "load_dataset_config": time_data.load_dataset_config,
        "save_window_predictions": time_saver.save_window_predictions,
        "get_available_terms": time_utils.get_available_terms,
    }


def resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_local_dataset(
    dataset_name: str, local_root: Path, allow_download: bool
) -> Path:
    if (local_root / dataset_name).exists():
        return local_root
    if not allow_download:
        raise FileNotFoundError(
            f"missing local TIME dataset {dataset_name}; rerun with --download to fetch it"
        )
    target = local_root
    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        local_dir=target,
        allow_patterns=[f"{dataset_name.split('/')[0]}/**"],
        max_workers=4,
    )
    if not (target / dataset_name).exists():
        raise FileNotFoundError(f"TIME download did not create {target / dataset_name}")
    return target


def make_context_batch(entries: list[dict], context_length: int) -> np.ndarray:
    import numpy as np

    arrays = [np.asarray(entry["target"], dtype=np.float32) for entry in entries]
    if any(value.ndim not in (1, 2) for value in arrays):
        raise ValueError(f"unexpected target shapes: {[value.shape for value in arrays]}")
    univariate = [value if value.ndim == 1 else value[0] for value in arrays]
    max_length = max(value.shape[0] for value in univariate)
    context = np.full((len(univariate), max_length), np.nan, dtype=np.float32)
    for row, value in enumerate(univariate):
        context[row, -value.shape[0]:] = value
        tail_start = max(0, context.shape[1] - context_length)
        tail = context[row, tail_start:]
        if int(np.isfinite(tail).sum()) < 8:
            observed = np.flatnonzero(np.isfinite(context[row]))
            fill = context[row, observed[-1]] if observed.size else 0.0
            tail[~np.isfinite(tail)] = fill
    return context


def forecast_horizon(
    model: object,
    context: np.ndarray,
    prediction_length: int,
) -> np.ndarray:
    """Roll the 48-step released head forward until the requested horizon."""
    import numpy as np

    prediction_length = int(prediction_length)
    if prediction_length < 1:
        raise ValueError(f"prediction_length must be positive, got {prediction_length}")
    full = np.empty((context.shape[0], prediction_length, len(model.quantiles)), dtype=np.float32)
    rolling = context.copy()
    median_index = len(model.quantiles) // 2
    done = 0
    chunk_size = model.horizon
    while done < prediction_length:
        predicted = model.forecast_batch(rolling)
        take = min(chunk_size, prediction_length - done)
        full[:, done:done + take] = predicted[:, :take]
        done += take
        if done < prediction_length:
            feedback = predicted[:, :take, median_index].astype(np.float32)
            rolling = np.concatenate((rolling, feedback), axis=1)[:, -model.context_length:]
    if not np.isfinite(full).all():
        raise RuntimeError("Fracast-0 produced a non-finite TIME forecast")
    return full


def run_configuration(
    args: argparse.Namespace,
    dataset_name: str,
    term: str,
    sources: dict,
) -> None:
    settings = sources["get_dataset_settings"](dataset_name, term, args.config)
    data_root = prepare_local_dataset(dataset_name, args.data_root, args.download)
    from gluonts.time_feature import get_seasonality

    probe = sources["Dataset"](name=dataset_name, term=term, storage_path=data_root)
    to_univariate = probe.target_dim != 1
    del probe
    dataset = sources["Dataset"](
        name=dataset_name,
        term=term,
        to_univariate=to_univariate,
        prediction_length=settings["prediction_length"],
        test_length=settings["test_length"],
        val_length=settings.get("val_length", 0),
        storage_path=data_root,
    )

    model = sources["FracastModel"].from_pretrained(
        args.weights_dir, weights=args.weights, device=args.device
    )
    test_inputs = list(dataset.test_data.input)
    output_batches: list[np.ndarray] = []
    batch_size = args.batch_size
    total = len(test_inputs)
    print(f"  inference: {total} instances, batch_size={batch_size}, device={args.device}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        context = make_context_batch(test_inputs[start:end], model.context_length)
        prediction = forecast_horizon(model, context, dataset.prediction_length)
        output_batches.append(prediction.transpose(0, 2, 1))
        print(f"    predicted {end}/{total}")
    fc_quantiles = np.concatenate(output_batches, axis=0)

    seasonality = get_seasonality(dataset.freq)
    metadata = sources["save_window_predictions"](
        dataset=dataset,
        fc_quantiles=fc_quantiles,
        ds_config=f"{dataset_name}/{term}",
        output_base_dir=args.output_dir,
        seasonality=seasonality,
        model_hyperparams={
            "model": MODEL_NAME,
            "context_length": model.context_length,
            "release_horizon": model.horizon,
            "rollout_chunks": math.ceil(dataset.prediction_length / model.horizon),
            "rollout_feedback": "median_quantile",
            "weights": args.weights,
        },
        quantile_levels=QUANTILE_LEVELS,
    )
    print(json.dumps({key: metadata[key] for key in (
        "num_series", "num_windows", "num_variates", "prediction_length"
    )}, sort_keys=True))


def validate_result(result_root: Path, dataset_name: str, term: str) -> dict:
    import numpy as np

    task_dir = result_root / dataset_name / term
    with np.load(task_dir / "metrics.npz") as metrics:
        summary = {name: float(np.nanmean(values)) for name, values in metrics.items()}
    with (task_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not math.isfinite(summary["MASE"]) or not math.isfinite(summary["CRPS"]):
        raise RuntimeError("TIME produced non-finite MASE or CRPS")
    return {
        "config": config,
        "MASE": summary["MASE"],
        "CRPS": summary["CRPS"],
        "MAE": summary["MAE"],
        "MSE": summary["MSE"],
    }


def has_valid_result(result_root: Path, dataset_name: str, term: str, weights: str) -> bool:
    task_dir = result_root / dataset_name / term
    if not all((task_dir / filename).is_file() for filename in (
        "config.json", "metrics.npz", "predictions.npz"
    )):
        return False
    try:
        with (task_dir / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("model") != MODEL_NAME or config.get("weights") != weights:
            return False
        validate_result(result_root, dataset_name, term)
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="TIME dataset key, e.g. ECDC_COVID/W")
    parser.add_argument("--terms", nargs="+", default=None, choices=("short", "medium", "long"))
    parser.add_argument(
        "--time-source",
        type=Path,
        required=True,
        help="pinned local clone of the TIME benchmark repository",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="TIME datasets.yaml path; defaults to the file in --time-source",
    )
    parser.add_argument(
        "--package-source",
        type=Path,
        required=True,
        help="local clone of the Fracast repository used by this run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/time"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--weights-dir", default="ztxtech/fracast-0")
    parser.add_argument("--weights", default="fp32", choices=("fp32", "w8"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    sources = load_sources(args)
    if args.config_path is None:
        args.config_path = (
            args.time_source / "src" / "timebench" / "config" / "datasets.yaml"
        )
    args.config = sources["load_dataset_config"](args.config_path)
    if args.dataset == "all_datasets":
        dataset_names = list(args.config.get("datasets", {}))
    else:
        dataset_names = [args.dataset]
    summary_path = args.output_dir / "progress.jsonl"
    completed = skipped = 0
    for dataset_name in dataset_names:
        terms = args.terms or sources["get_available_terms"](dataset_name, args.config)
        if not terms:
            raise ValueError(f"no TIME terms configured for {dataset_name}")
        for term in terms:
            task = f"{dataset_name}/{term}"
            if args.skip_existing and has_valid_result(
                args.output_dir, dataset_name, term, args.weights
            ):
                print(f"[{task}] SKIP valid result")
                skipped += 1
                continue
            print(f"[{task}]")
            run_configuration(args, dataset_name, term, sources)
            result = validate_result(args.output_dir, dataset_name, term)
            completed += 1
            record = {
                "task": task,
                "completed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "MASE": result["MASE"],
                "CRPS": result["CRPS"],
            }
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, indent=2, sort_keys=True))
    print(f"progress: completed={completed}, skipped={skipped}")


if __name__ == "__main__":
    main()
