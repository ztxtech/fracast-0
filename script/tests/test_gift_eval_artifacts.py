"""CPU checks for the archived GIFT-Eval submission artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "script" / "gift_eval"
RESULTS = ARCHIVE / "results" / "Fracast-0"
MANIFEST = ARCHIVE / "run-manifest.json"

REQUIRED_FILES = [
    ARCHIVE / "readme.md",
    ARCHIVE / "evaluate.py",
    ARCHIVE / "analysis" / "protocol_summary.csv",
    RESULTS / "all_results.csv",
    RESULTS / "config.json",
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

AGGREGATES = {
    "short": (0.769573, 0.568038),
    "medium": (0.842959, 0.557024),
    "long": (0.875558, 0.555953),
    "overall": (0.807133, 0.563008),
}


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not path.is_file() or path.is_symlink()]
    assert not missing, f"missing or non-regular GIFT-Eval archive files: {missing}"

    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--error-unmatch",
            str((RESULTS / "all_results.csv").relative_to(ROOT)),
        ],
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, "archived submission is not tracked by Git"

    public_files = sorted(
        path
        for path in ARCHIVE.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".json", ".md", ".py"}
    )
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        for marker in LOCAL_PATH_MARKERS:
            assert marker not in text, f"{path.relative_to(ROOT)} contains {marker!r}"

    subprocess.run(
        [sys.executable, "-m", "py_compile", ARCHIVE / "evaluate.py"],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ARCHIVE / "evaluate.py"), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["official_results"].endswith(
        "gift-eval/tree/main/results/Fracast-0"
    )
    assert manifest["evaluation"]["configurations_completed"] == 97
    assert manifest["evaluation"]["configurations_total"] == 97
    assert manifest["evaluation"]["task_failures"] == 0
    assert manifest["results"]["overall"]["normalized_mase"] == 0.807133

    for relative, expected in manifest["artifacts"].items():
        actual = hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest()
        assert actual == expected, f"hash mismatch for {relative}: {actual}"

    with (RESULTS / "all_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 97, f"expected 97 GIFT-Eval configurations, got {len(rows)}"
    assert len({row["dataset"] for row in rows}) == 97, "configuration identifiers must be unique"
    assert all(row["model"] == "Fracast-0" for row in rows)
    numeric_columns = list(rows[0])[2:13]
    assert len(numeric_columns) == 11
    assert all(all(row[column] for column in numeric_columns) for row in rows)

    with (ARCHIVE / "analysis" / "protocol_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = {row["protocol_split"]: row for row in csv.DictReader(handle)}
    assert set(summary) == set(AGGREGATES)
    for split, (mase, mwql) in AGGREGATES.items():
        assert float(summary[split]["normalized_mase"]) == mase
        assert float(summary[split]["normalized_mwql"]) == mwql

    submission = json.loads((RESULTS / "config.json").read_text(encoding="utf-8"))
    assert submission["model"] == "Fracast-0"
    assert submission["model_type"] == "pretrained"
    assert submission["model_dtype"] == "float32"
    assert submission["testdata_leakage"] == "Yes"

    print("[PASS] GIFT-Eval archive files, metadata, results, and disclosure checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
