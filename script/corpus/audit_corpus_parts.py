"""Audit corpus parts independently from on-disk shard arrays.

Reports and per-part manifests can miss a missing part or an identifier/name
mismatch. This audit recomputes partition assignment and shard invariants.

Run from the repository root.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tmp", "corpus_report")

DEFAULT_CORPORA = {
    "pret": ("data/pretrain_full", 32, "data/corpus_fast/pret"),
    "lotsa": ("data/lotsa_full", 32, "data/corpus_fast/lotsa"),
    "chronos": ("data/chronos_full", 32, "data/corpus_fast/chronos"),
    "boom": ("data/boom", 8, "data/corpus_fast/boom"),
    "fev": ("data/fev", 4, "data/corpus_fast/fev"),
}

_PART_RE = re.compile(r"^p(\d+)$")
_SPLIT_RE = re.compile(r"^p(\d+)_split\d+$")


def npy_data_bytes(path: str) -> int:
    """Return the payload byte count of an .npy file, excluding its header."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        header_length = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - header_length


def lpt_pack(source_dir: str, part_count: int) -> list[list[str]]:
    """Independently rerun byte-balanced LPT packing for Arrow/Parquet corpora."""
    datasets: list[tuple[str, int]] = []
    for name in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue

        has_data = False
        total_bytes = 0
        for root, _dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith((".arrow", ".parquet")):
                    has_data = True
                try:
                    total_bytes += os.path.getsize(os.path.join(root, filename))
                except OSError:
                    pass
        if has_data:
            datasets.append((name, total_bytes))

    sizes = dict(datasets)
    buckets: list[list[str]] = [[] for _ in range(part_count)]
    loads = [0] * part_count
    for name in sorted(sizes, key=lambda item: -sizes[item]):
        target = loads.index(min(loads))
        buckets[target].append(name)
        loads[target] += sizes[name]
    return buckets


def read_names(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def check_part(part_dir: str, expected: list[str] | None) -> dict:
    """Audit one part directory and return its detailed status."""
    name = os.path.basename(part_dir)
    datasets = read_names(os.path.join(part_dir, "datasets.txt"))
    freqs = read_names(os.path.join(part_dir, "freqs.txt"))
    shards = sorted(glob.glob(os.path.join(part_dir, "shard*.values.f32.npy")))
    problems: list[str] = []
    n_rows = n_points = 0
    max_dataset = max_frequency = -1

    for values_path in shards:
        base = values_path[: -len(".values.f32.npy")]
        try:
            lengths = np.load(base + ".lengths.npy")
            offsets = np.load(base + ".offsets.npy")
            dataset_ids = np.load(base + ".ds_id.npy")
            frequency_ids = np.load(base + ".freq_id.npy")
            timestamps = np.load(base + ".ts.npy")
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"{os.path.basename(values_path)}: cannot load arrays "
                f"({type(exc).__name__})"
            )
            continue

        if offsets.size != lengths.size + 1:
            problems.append(
                f"{os.path.basename(values_path)}: offsets length {offsets.size} "
                f"!= lengths {lengths.size} + 1"
            )
        elif int(offsets[-1]) != int(lengths.sum()):
            problems.append(
                f"{os.path.basename(values_path)}: final offset {int(offsets[-1])} "
                f"!= length sum {int(lengths.sum())}"
            )
        if (dataset_ids.size != lengths.size or frequency_ids.size != lengths.size
                or timestamps.size != lengths.size):
            problems.append(
                f"{os.path.basename(values_path)}: ds/freq/ts lengths differ from lengths"
            )
        if npy_data_bytes(values_path) != 4 * int(lengths.sum()):
            problems.append(
                f"{os.path.basename(values_path)}: payload bytes "
                f"{npy_data_bytes(values_path)} != {4 * int(lengths.sum())}"
            )

        n_rows += int(lengths.size)
        n_points += int(lengths.sum())
        if dataset_ids.size:
            max_dataset = max(max_dataset, int(dataset_ids.max()))
        if frequency_ids.size:
            max_frequency = max(max_frequency, int(frequency_ids.max()))

    if datasets and max_dataset >= len(datasets):
        problems.append(
            f"ds_id out of range: max={max_dataset}; datasets.txt has {len(datasets)}"
        )
    if freqs and max_frequency >= len(freqs):
        problems.append(
            f"freq_id out of range: max={max_frequency}; freqs.txt has {len(freqs)}"
        )
    if expected is not None and sorted(datasets) != sorted(expected):
        problems.append("part assignment differs from independently recomputed LPT packing")

    return {
        "part": name,
        "n_shards": len(shards),
        "n_rows": n_rows,
        "n_points": n_points,
        "n_datasets": len(datasets),
        "has_manifest": os.path.exists(os.path.join(part_dir, "manifest.json")),
        "has_ds_names": os.path.exists(os.path.join(part_dir, "ds_names.json")),
        "datasets": datasets,
        "expected": expected,
        "problems": problems,
        "_dir": part_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", action="append", default=[],
        help="name=source:parts; repeatable; omit to use the --all defaults",
    )
    parser.add_argument("--all", action="store_true", help="audit all public corpora in readme")
    parser.add_argument(
        "--write-ds-names", action="store_true", help="write missing ds_names.json files"
    )
    args = parser.parse_args()

    specs = dict(DEFAULT_CORPORA)
    if args.corpus:
        specs = {}
        for item in args.corpus:
            name, remainder = item.split("=", 1)
            source, parts = remainder.rsplit(":", 1)
            specs[name] = (source, int(parts), f"data/corpus_fast/{name}")

    os.makedirs(OUT, exist_ok=True)
    report: dict = {}
    return_code = 0
    for name, (source, part_count, corpus_dir) in specs.items():
        source_dir = os.path.join(ROOT, source)
        corpus_path = os.path.join(ROOT, corpus_dir)
        if not os.path.isdir(corpus_path):
            print(f"[{name}] corpus directory does not exist: {corpus_dir}")
            continue
        packing = lpt_pack(source_dir, part_count) if os.path.isdir(source_dir) else None
        if packing is None:
            print(
                f"[{name}] source directory does not exist; "
                f"assignment check skipped: {source}"
            )

        part_dirs = [
            path for path in sorted(glob.glob(os.path.join(corpus_path, "p*")))
            if os.path.isdir(path)
        ]
        entries = []
        seen_numbers = set()
        split_dir_count = 0
        for part_dir in part_dirs:
            basename = os.path.basename(part_dir)
            split_match = _SPLIT_RE.fullmatch(basename)
            if split_match is not None:
                split_dir_count += 1
                seen_numbers.add(int(split_match.group(1)))
                entries.append(check_part(part_dir, None))
                continue

            part_match = _PART_RE.fullmatch(basename)
            expected = None
            if part_match is not None and packing is not None:
                number = int(part_match.group(1))
                seen_numbers.add(number)
                expected = packing[number] if number < len(packing) else None
            entries.append(check_part(part_dir, expected))

        missing = (
            sorted(set(range(part_count)) - seen_numbers) if packing is not None else []
        )
        merged_names = read_names(os.path.join(corpus_path, "datasets.txt"))
        coverage = None
        if merged_names and packing is not None:
            source_names = sorted({name for bucket in packing for name in bucket})
            coverage = {
                "src": len(source_names),
                "merged": len(merged_names),
                "missing": sorted(set(source_names) - set(merged_names)),
                "extra": sorted(set(merged_names) - set(source_names)),
            }
            if coverage["missing"] or coverage["extra"]:
                print(
                    f"        [coverage] source {coverage['src']} vs merged "
                    f"{coverage['merged']}: missing={coverage['missing']} "
                    f"extra={coverage['extra']}"
                )

        complete = [
            entry for entry in entries
            if entry["has_manifest"] and not entry["problems"]
        ]
        partial = [entry for entry in entries if not entry["has_manifest"]]
        broken = [entry for entry in entries if entry["problems"]]
        total_rows = sum(entry["n_rows"] for entry in entries)
        total_points = sum(entry["n_points"] for entry in entries)
        split_text = f" (+{split_dir_count} split directories)" if split_dir_count else ""
        print(
            f"[{name}] parts {len(seen_numbers)}/{part_count} | directories "
            f"{len(entries)}{split_text} | complete {len(complete)} | partial "
            f"{len(partial)} | problems {len(broken)} | missing "
            f"{missing if missing else 'none'}"
        )
        print(f"        total {total_rows:,} rows / {total_points / 1e9:.2f}B points")
        if coverage is not None:
            status = "PASS" if not (coverage["missing"] or coverage["extra"]) else "FAIL"
            print(
                f"        dataset coverage: source {coverage['src']} / merged "
                f"{coverage['merged']} {status}"
            )
        for entry in partial:
            print(
                f"        [partial] {entry['part']}: {entry['n_shards']} shards / "
                f"{entry['n_rows']:,} rows; manifest missing"
            )
        for entry in broken:
            for problem in entry["problems"]:
                print(f"        [problem] {entry['part']}: {problem}")

        if args.write_ds_names:
            for entry in entries:
                if entry["has_ds_names"] or not entry["datasets"] or entry["problems"]:
                    continue
                output_path = os.path.join(entry["_dir"], "ds_names.json")
                with open(output_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"part": entry["part"], "datasets": entry["datasets"]},
                        fh,
                        ensure_ascii=False,
                        indent=1,
                    )
                print(
                    f"        [write] {entry['part']}/ds_names.json "
                    f"({len(entry['datasets'])} datasets)"
                )

        report[name] = {
            "src": source,
            "parts": part_count,
            "n_part_dirs": len(entries),
            "n_split_dirs": split_dir_count,
            "missing": missing,
            "coverage": coverage,
            "complete": len(complete),
            "partial": len(partial),
            "broken": len(broken),
            "n_rows": total_rows,
            "n_points": total_points,
            "entries": [dict(entry) for entry in entries],
        }
        if missing or partial or broken:
            return_code = 1

    with open(os.path.join(OUT, "parts_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    failed = any(
        value["missing"] or value["partial"] or value["broken"]
        for value in report.values()
    )
    print(
        f"\nreport -> {os.path.relpath(os.path.join(OUT, 'parts_audit.json'), ROOT)} "
        f"({'FAIL: review details above' if failed else 'PASS'})"
    )
    sys.exit(return_code)


if __name__ == "__main__":
    main()
