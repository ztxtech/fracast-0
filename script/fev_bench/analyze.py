"""Regenerate the archived FEV-Bench raw and leakage-controlled rankings."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import fev
import pandas as pd


PINNED_COMMIT = "81cf1255bb0c88dc039ae9bca23f73db6d9dfa61"
BASELINE_MODEL = "Seasonal Naive"
LEAKAGE_IMPUTATION_MODEL = "Chronos-Bolt"
METRICS = ["SQL", "MASE", "WQL", "WAPE"]
MODES = [("raw", None), ("controlled", LEAKAGE_IMPUTATION_MODEL)]
EXCLUDED_MODELS = ["Toto-2.0-4m", "Toto-2.0-313m", "Toto-2.0-1B"]

LEADERBOARD_COLUMNS = [
    "model_name",
    "metric",
    "mode",
    "win_rate",
    "skill_score",
    "median_training_time_s_per100",
    "median_inference_time_s_per100",
    "median_e2e_time_s_per100",
    "training_corpus_overlap",
    "num_failures",
    "win_rate_pct",
    "skill_score_pct",
    "rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fev-repo", required=True, type=Path)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    return parser.parse_args()


def check_pinned_commit(repository: Path) -> None:
    actual = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != PINNED_COMMIT:
        raise ValueError(f"FEV repository must be at {PINNED_COMMIT}, got {actual}")


def load_summaries(repository: Path, result: Path) -> pd.DataFrame:
    official_files = sorted((repository / "benchmarks" / "fev_bench" / "results").glob("*.csv"))
    if not official_files:
        raise FileNotFoundError(f"no official results in {repository}")
    frames = [pd.read_csv(path) for path in official_files]
    frames.append(pd.read_csv(result))
    return pd.concat(frames, ignore_index=True)


def compute_rank(summaries: pd.DataFrame, metric: str, mode: str, imputation_model: str | None) -> pd.DataFrame:
    board = fev.analysis.leaderboard(
        summaries=summaries,
        metric_column=metric,
        missing_strategy="impute",
        baseline_model=BASELINE_MODEL,
        included_models=None,
        excluded_models=None,
        leakage_imputation_model=imputation_model,
        n_resamples=None,
        normalize_time_per_n_forecasts=100,
    )
    rows = board.drop(index=EXCLUDED_MODELS).reset_index()
    rows["metric"] = metric
    rows["mode"] = mode
    rows["win_rate_pct"] = rows["win_rate"] * 100.0
    rows["skill_score_pct"] = rows["skill_score"] * 100.0
    rows["rank"] = rows["win_rate"].rank(method="min", ascending=False).astype(int)
    return rows[LEADERBOARD_COLUMNS]


def write_fracast_ranks(tables: dict[tuple[str, str], pd.DataFrame], output_dir: Path) -> None:
    columns = [
        "metric",
        "mode",
        "rank",
        "models_visible",
        "win_rate_pct",
        "skill_score_pct",
        "training_corpus_overlap_pct",
        "num_failures",
        "median_inference_time_s_per100",
    ]
    records = []
    for metric in METRICS:
        for mode, _ in MODES:
            row = tables[(metric, mode)]
            selected = row.loc[row["model_name"] == "fracast-0"].iloc[0]
            records.append(
                {
                    "metric": metric,
                    "mode": mode,
                    "rank": int(selected["rank"]),
                    "models_visible": len(row),
                    "win_rate_pct": float(selected["win_rate_pct"]),
                    "skill_score_pct": float(selected["skill_score_pct"]),
                    "training_corpus_overlap_pct": float(selected["training_corpus_overlap"]) * 100.0,
                    "num_failures": int(selected["num_failures"]),
                    "median_inference_time_s_per100": float(selected["median_inference_time_s_per100"]),
                }
            )
    pd.DataFrame(records, columns=columns).to_csv(output_dir / "fracast_ranks.csv", index=False)


def main() -> int:
    args = parse_args()
    result = args.result or Path(__file__).resolve().parent / "results" / "fracast-0.csv"
    check_pinned_commit(args.fev_repo)
    summaries = load_summaries(args.fev_repo, result)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for metric in METRICS:
        for mode, imputation_model in MODES:
            table = compute_rank(summaries, metric, mode, imputation_model)
            tables[(metric, mode)] = table
            table.to_csv(args.out_dir / f"leaderboard_{mode}_{metric}.csv", index=False)
    write_fracast_ranks(tables, args.out_dir)
    print(f"wrote {len(tables) + 1} ranking tables to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
