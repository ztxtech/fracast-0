"""CPU checks for the archived FEV-Bench evaluation artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "script" / "fev_bench"
RESULTS = ARCHIVE / "results" / "fracast-0.csv"
MANIFEST = ARCHIVE / "run-manifest.json"

REQUIRED_FILES = [
    ARCHIVE / "README.md",
    ARCHIVE / "requirements.txt",
    ARCHIVE / "reproduce.sh",
    ARCHIVE / "analyze.py",
    ARCHIVE / "models" / "fracast-0" / "model.py",
    ARCHIVE / "models" / "fracast-0" / "requirements.txt",
    RESULTS,
    MANIFEST,
    ARCHIVE / "analysis" / "fracast_ranks.csv",
]

LOCAL_PATH_MARKERS = [
    "/Users/",
    "/private/tmp/",
    "/tmp/",
    "Documents/code/",
    "/share_data/",
    "kangaroo/",
]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not path.is_file() or path.is_symlink()]
    assert not missing, f"missing or non-regular FEV archive files: {missing}"
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(RESULTS.relative_to(ROOT))],
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, f"archive result is not tracked by Git: {RESULTS}"

    public_files = sorted(
        path
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".sh", ".json", ".csv", ".txt"}
    )
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for marker in LOCAL_PATH_MARKERS:
            assert marker not in text, f"{path.relative_to(ROOT)} contains local path marker {marker!r}"

    reproduce_text = (ARCHIVE / "reproduce.sh").read_text(encoding="utf-8")
    assert "MODEL_KWARGS=$(printf" in reproduce_text
    assert '"batch_size":%s' in reproduce_text
    assert '"device":"%s"' in reproduce_text
    assert "--model-kwargs \"$MODEL_KWARGS\"" in reproduce_text

    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100, f"expected 100 FEV tasks, got {len(rows)}"
    assert len({row["task_name"] for row in rows}) == 100, "FEV task names must be unique"
    for metric in ("SQL", "MASE"):
        assert all(row[metric] for row in rows), f"missing required metric {metric}"
    assert all(row["trained_on_this_dataset"] == "True" for row in rows)
    assert all('"/Users/' not in row["model_kwargs"] for row in rows)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["official_repository"] == "autogluon/fev"
    assert manifest["official_commit"] == "81cf1255bb0c88dc039ae9bca23f73db6d9dfa61"
    assert manifest["evaluation"]["tasks_completed"] == 100
    assert manifest["evaluation"]["task_failures"] == 0
    assert manifest["disclosure"]["training_corpus_overlap_pct"] == 100.0

    archive_root = ARCHIVE
    for relative, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((archive_root / relative).read_bytes()).hexdigest()
        assert actual == expected, f"hash mismatch for {relative}: {actual}"

    ranks_path = ARCHIVE / "analysis" / "fracast_ranks.csv"
    with ranks_path.open(newline="", encoding="utf-8") as handle:
        ranks = {(row["metric"], row["mode"]): row for row in csv.DictReader(handle)}
    assert len(ranks) == 8
    assert ranks[("SQL", "raw")]["rank"] == "17"
    assert ranks[("SQL", "controlled")]["rank"] == "16"
    assert ranks[("MASE", "raw")]["rank"] == "20"

    print("[PASS] FEV-Bench archive files, metadata, and disclosure checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
