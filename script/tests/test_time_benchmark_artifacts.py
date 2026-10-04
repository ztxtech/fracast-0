"""Local checks for the archived TIME benchmark artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "script" / "time_benchmark"
MANIFEST = ARCHIVE / "run-manifest.json"

REQUIRED_FILES = [
    ARCHIVE / "README.md",
    ARCHIVE / "requirements.txt",
    ARCHIVE / "analyze_pareto.py",
    ARCHIVE / "run_fracast.py",
    ARCHIVE / "validate_submission.py",
    ARCHIVE / "submission" / "Fracast-0" / "README.md",
    ARCHIVE / "analysis" / "TIME_overall_leaderboard.csv",
    ARCHIVE / "analysis" / "TIME_horizon_leaderboard.csv",
    ARCHIVE / "analysis" / "Fracast-0_TIME_tasks.csv",
    ARCHIVE / "analysis" / "TIME_pareto.csv",
    ARCHIVE / "analysis" / "TIME_pareto_summary.json",
    MANIFEST,
]

LOCAL_PATH_MARKERS = [
    "/Users/",
    "/private/tmp/",
    "/tmp/",
    "Documents/code/",
    "/share_data/",
    "kangaroo/",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not path.is_file() or path.is_symlink()]
    assert not missing, f"missing or non-regular TIME archive files: {missing}"

    public_files = sorted(
        path
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".json", ".csv", ".txt"}
    )
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for marker in LOCAL_PATH_MARKERS:
            assert marker not in text, f"{path.relative_to(ROOT)} contains {marker!r}"

    subprocess.run(
        [sys.executable, "-m", "py_compile", *(path for path in ARCHIVE.glob("*.py"))],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ARCHIVE / "run_fracast.py"), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, str(ARCHIVE / "validate_submission.py"), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["benchmark"]["repository_commit"] == (
        "3ca5c41d71c76f3c70632c0121142d325350ee32"
    )
    assert manifest["evaluation"]["tasks_completed"] == 98
    assert manifest["evaluation"]["rollout_tasks"] == 47
    assert manifest["official_output_repository"] == (
        "https://huggingface.co/datasets/Real-TSF/TIME-Output"
    )
    assert manifest["disclosures"][0] == (
        "47 of 98 tasks use the released 48-step head with median-quantile rollout."
    )

    expected_hash = hashlib.sha256(
        (ROOT / "weights" / "model.safetensors").read_bytes()
    ).hexdigest()
    assert manifest["model"]["weights_sha256"] == expected_hash

    for relative, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest()
        assert actual == expected, f"hash mismatch for {relative}: {actual}"

    overall = {row["model"]: row for row in read_csv(
        ARCHIVE / "analysis" / "TIME_overall_leaderboard.csv"
    )}
    assert len(overall) == 29
    fracast = overall["Fracast-0"]
    assert fracast["leaderboard_rank"] == "24"
    assert float(fracast["MASE_norm"]) == 0.767965
    assert float(fracast["CRPS_norm"]) == 0.649192

    tasks = read_csv(ARCHIVE / "analysis" / "Fracast-0_TIME_tasks.csv")
    assert len(tasks) == 98
    assert len({(row["dataset_id"], row["horizon"]) for row in tasks}) == 98
    assert {row["horizon"] for row in tasks} == {"short", "medium", "long"}

    horizons = {
        (row["model"], row["horizon"]): row
        for row in read_csv(ARCHIVE / "analysis" / "TIME_horizon_leaderboard.csv")
    }
    assert len(horizons) == 87
    expected_horizons = {
        "short": (0.701284, 0.586018),
        "medium": (0.855065, 0.736344),
        "long": (0.833422, 0.708427),
    }
    for horizon, (mase, crps) in expected_horizons.items():
        row = horizons[("Fracast-0", horizon)]
        assert float(row["MASE_norm"]) == mase
        assert float(row["CRPS_norm"]) == crps

    pareto = {row["model"]: row for row in read_csv(
        ARCHIVE / "analysis" / "TIME_pareto.csv"
    )}
    assert len(pareto) == 29
    assert pareto["Fracast-0"]["parameter_count"] == "85001"
    assert pareto["Fracast-0"]["pareto_parameter_mase"] == "True"
    assert pareto["Fracast-0"]["pareto_parameter_crps"] == "True"
    assert pareto["Fracast-0"]["parameter_kind"] == "exact"

    analysis = ARCHIVE / "analysis"
    before = (
        (analysis / "TIME_pareto.csv").read_bytes(),
        (analysis / "TIME_pareto_summary.json").read_bytes(),
    )
    subprocess.run([sys.executable, str(ARCHIVE / "analyze_pareto.py")], check=True)
    after = (
        (analysis / "TIME_pareto.csv").read_bytes(),
        (analysis / "TIME_pareto_summary.json").read_bytes(),
    )
    assert before == after, "Pareto regeneration is not byte-identical"

    print("[PASS] TIME archive files, metadata, scores, Pareto, and disclosure checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
