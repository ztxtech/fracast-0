"""Reconcile local public corpora against official Hugging Face manifests.

File counts alone do not prove integrity. This first pass compares datasets
by name and reports local file-count and byte-size gaps. A second pass
recomputes corpus contents in audit_corpus_bytes.py; an independent review
can then sample original Arrow records.

Usage:
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus.py \
        --repo Salesforce/GiftEvalPretrain --dir data/pretrain_full
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus.py --all
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

# Keep all data inside the repository-local data directory.
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
REPOS = {
    "pret": ("Salesforce/GiftEvalPretrain", os.path.join(DATA, "pretrain_full")),
    "lotsa": ("Salesforce/lotsa_data", os.path.join(DATA, "lotsa_full")),
    "chronos": ("autogluon/chronos_datasets", os.path.join(DATA, "chronos_full")),
}


def local_stats(d: str):
    """Collect local file counts and byte totals per first-level dataset."""
    out = defaultdict(lambda: [0, 0])
    if not os.path.isdir(d):
        return out
    for root, _dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        if rel == ".":
            top = "(root)"
        else:
            top = rel.split(os.sep)[0]
        for f in files:
            if f.endswith((".lock", ".incomplete")) or f.startswith("."):
                continue
            p = os.path.join(root, f)
            try:
                out[top][0] += 1
                out[top][1] += os.path.getsize(p)
            except OSError:
                pass
    return out


def official_stats(repo: str):
    """Collect official file counts and byte totals per first-level dataset."""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo, files_metadata=True)
    out = defaultdict(lambda: [0, 0])
    for f in info.siblings:
        name = f.rfilename
        if name.startswith("."):
            continue
        top = name.split("/")[0] if "/" in name else "(root)"
        out[top][0] += 1
        out[top][1] += f.size or 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", default=None, help="Write the report to a JSON file")
    a = ap.parse_args()

    targets = []
    if a.all:
        targets = [(k, r, d) for k, (r, d) in REPOS.items()]
    elif a.repo and a.dir:
        targets = [("custom", a.repo, a.dir)]
    else:
        ap.error("specify --repo and --dir, or use --all")

    report = {}
    for tag, repo, d in targets:
        off = official_stats(repo)
        loc = local_stats(d)
        n_off_files = sum(v[0] for v in off.values())
        n_loc_files = sum(v[0] for v in loc.values())
        b_off = sum(v[1] for v in off.values())
        b_loc = sum(v[1] for v in loc.values())
        print(f"\n=== {tag}: {repo}")
        print(f"  official: {len(off):3d} datasets / {n_off_files:5d} files / {b_off/1e9:7.1f} GB")
        print(f"  local:    {len(loc):3d} datasets / {n_loc_files:5d} files / {b_loc/1e9:7.1f} GB"
              f"  complete={100*b_loc/max(b_off,1):5.1f}% by bytes")
        # Compare datasets individually and rank the largest byte deficits.
        diff = []
        for ds, (nf, nb) in sorted(off.items()):
            lf, lb = loc.get(ds, [0, 0])
            if lb < nb * 0.999:  # Allow 0.1 percent for filesystem metadata.
                diff.append((ds, nb, lb, nf, lf))
        diff.sort(key=lambda x: -(x[1] - x[2]))
        print(f"  incomplete datasets: {len(diff)}/{len(off)}")
        for ds, nb, lb, nf, lf in diff[:12]:
            print(f"    {ds:42s} {lb/1e9:7.2f}/{nb/1e9:7.2f} GB  ({lf}/{nf} files)"
                  f"  missing {(nb-lb)/1e9:6.2f} GB")
        if len(diff) > 12:
            print(f"    ... {len(diff)-12} additional incomplete datasets")
        report[tag] = {
            "repo": repo, "dir": d,
            "official": {"datasets": len(off), "files": n_off_files, "bytes": b_off},
            "local": {"datasets": len(loc), "files": n_loc_files, "bytes": b_loc},
            "incomplete": [{"dataset": ds, "official_bytes": nb, "local_bytes": lb,
                            "official_files": nf, "local_files": lf} for ds, nb, lb, nf, lf in diff],
        }
        missing_ds = sorted(set(off) - set(loc))
        if missing_ds:
            print(f"  completely missing datasets ({len(missing_ds)}): {missing_ds[:10]}")

    if a.json:
        json.dump(report, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"\nPASS: report written to {a.json}")
    print("\nA dataset is complete when local bytes are at least 99.9 percent of official bytes.")
    print("Run byte-level and independent sequence audits after this reconciliation.")


if __name__ == "__main__":
    main()
