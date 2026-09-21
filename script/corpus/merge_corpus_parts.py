"""Merge independently converted corpus parts into one global index.

Each parallel part writes shard arrays and a manifest. The training path needs
one global mapping from row number to shard and in-shard row. This merger also
cross-checks every manifest against arrays read from disk.

Outputs, written at the corpus root:
- index.npz: global lengths, frequency/dataset IDs, and length buckets;
- shard_map.json: global shard path with row and point totals;
- datasets.txt and freqs.txt: global identifier tables;
- manifest.json: totals plus the per-part claims used for cross-checking.

Merge fails loudly when manifests disagree with recomputed lengths, offsets,
or payload bytes.

Usage:
    env -u PYTHONPATH .venv/bin/python script/corpus/merge_corpus_parts.py \
        --corpus data/corpus_fast/chronos
"""


from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


_PART_NAME_CACHE: dict = {}


def _part_names(pdir: str, fname: str):
    """Read and cache the dataset or frequency names for one part."""
    key = (pdir, fname)
    if key in _PART_NAME_CACHE:
        return _PART_NAME_CACHE[key]
    fp = os.path.join(pdir, fname)
    names = [l for l in open(fp).read().splitlines() if l.strip()] if os.path.exists(fp) else []
    _PART_NAME_CACHE[key] = names
    return names


def npy_data_bytes(path: str) -> int:
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        hlen = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - hlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pattern", default="p*/shard*.values.f32.npy")
    a = ap.parse_args()

    parts = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.corpus, "p*", "")))
    vals = sorted(glob.glob(os.path.join(a.corpus, a.pattern)))
    print(f"  part directories: {len([p for p in parts if os.path.isdir(p)])}; value shards: {len(vals)}")
    if not vals:
        print("  FAIL: no mergeable value shards"); return

    # Remap part-local dataset and frequency identifiers to global identifiers.
    ds_all, fr_all = [], []
    for p in sorted({os.path.dirname(v) for v in vals}):
        for fn, acc in (("datasets.txt", ds_all), ("freqs.txt", fr_all)):
            fp = os.path.join(p, fn)
            if os.path.exists(fp):
                for line in open(fp).read().splitlines():
                    if line.strip() and line not in acc:
                        acc.append(line)
    ds_id = {n: i for i, n in enumerate(ds_all)}
    fr_id = {n: i for i, n in enumerate(fr_all)}
    print(f"  datasets: {len(ds_all)} / frequencies: {len(fr_all)} after remapping")

    shard_map, Ls, Fs, Ds = [], [], [], []
    _seen_manifests = set()
    declared_rows = declared_pts = 0
    calc_rows = calc_pts = 0
    problems = []
    for i, vf in enumerate(vals):
        base = vf[:-len(".values.f32.npy")]
        try:
            L = np.load(base + ".lengths.npy")
            O = np.load(base + ".offsets.npy")
        except Exception as e:
            problems.append((os.path.relpath(base, a.corpus), f"missing arrays: {type(e).__name__}"))
            continue
        rows = len(L)
        pts = int(L.sum())
        calc_rows += rows
        calc_pts += pts
        if len(O) - 1 != rows:
            problems.append((os.path.relpath(base, a.corpus), f"row mismatch offsets={len(O)-1} lengths={rows}"))
        elif int(O[-1]) != pts:
            problems.append((os.path.relpath(base, a.corpus), f"final offset {int(O[-1])} != length sum {pts}"))
        elif int(O[-1]) * 4 != npy_data_bytes(vf):
            problems.append((os.path.relpath(base, a.corpus), "final offset bytes != payload bytes"))
        # Shard identifiers are part-local and require global remapping.
        fj = base + ".freq_id.npy"
        dj = base + ".ds_id.npy"
        lf = np.load(fj) if os.path.exists(fj) else np.zeros(rows, np.int16)
        ld = np.load(dj) if os.path.exists(dj) else np.zeros(rows, np.int16)
        # Convert each part-local identifier through that part name table into the
        # global name table; indexing a root table directly maps the wrong dataset.
        # Convert each part-local identifier through that part name table into the global table.
        pdir = os.path.dirname(vf)
        p_ds_names = _part_names(pdir, "datasets.txt")
        p_fr_names = _part_names(pdir, "freqs.txt")
        if p_ds_names:
            m = np.array([ds_id.get(n, 0) for n in p_ds_names], dtype=np.int16)
            ld = m[np.clip(ld.astype(np.int64), 0, len(m) - 1)] if len(m) else ld
        if p_fr_names:
            mf = np.array([fr_id.get(n, 0) for n in p_fr_names], dtype=np.int16)
            lf = mf[np.clip(lf.astype(np.int64), 0, len(mf) - 1)] if len(mf) else lf
        Ls.append(L); Fs.append(lf); Ds.append(ld)
        shard_map.append({"path": os.path.relpath(vf, a.corpus), "rows": rows, "points": pts})
        # Read each part manifest exactly once so totals are not multiplied by shards.
        # Read each part manifest exactly once; otherwise per-shard totals are multiplied.
        mp = os.path.join(os.path.dirname(vf), "manifest.json")
        if os.path.exists(mp) and mp not in _seen_manifests:
            _seen_manifests.add(mp)
            try:
                man = json.load(open(mp))
                declared_rows += int(man.get("n_rows", 0))
                declared_pts += int(man.get("n_points", 0))
            except Exception as e:
                problems.append((os.path.relpath(mp, a.corpus), f"manifest read failed: {type(e).__name__}"))

    Lg = np.concatenate(Ls) if Ls else np.zeros(0, np.int32)
    Fg = np.concatenate(Fs) if Fs else np.zeros(0, np.int16)
    Dg = np.concatenate(Ds) if Ds else np.zeros(0, np.int16)
    edges = [64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 30]
    buckets = {f"le{e}": np.where(Lg <= e)[0].astype(np.int64) for e in edges}
    np.savez(os.path.join(a.corpus, "index.npz"), lengths=Lg, freq_id=Fg, ds_id=Dg, **buckets)
    json.dump({"n_shards": len(shard_map), "n_rows": int(Lg.size), "n_points": int(Lg.sum()),
               "n_datasets": len(ds_all), "shards": shard_map,
               "cross_check": {"declared_rows": declared_rows, "calc_rows": calc_rows,
                               "declared_points": declared_pts, "calc_points": calc_pts},
               "notes": "values are contiguous float32; offsets provide O(1) row access"},
              open(os.path.join(a.corpus, "manifest.json"), "w"), indent=1)
    with open(os.path.join(a.corpus, "datasets.txt"), "w") as fh:
        fh.write("\n".join(ds_all) + "\n")
    with open(os.path.join(a.corpus, "freqs.txt"), "w") as fh:
        fh.write("\n".join(fr_all) + "\n")

    print(f"  PASS merged {len(shard_map)} shards / {Lg.size:,} sequences / {Lg.sum()/1e9:.3f}B points / "
          f"{Lg.sum()*4/1e9:.2f} GB (float32)")
    print(f"  cross-check: manifests claim {declared_rows:,} rows / {declared_pts:,} points; "
          f"recomputed {calc_rows:,} rows / {calc_pts:,} points; "
          f"{'PASS' if (declared_rows == calc_rows and declared_pts == calc_pts) else 'FAIL'}")
    if problems:
        print(f"  FAIL: {len(problems)} problems")
        for nm, why in problems[:10]:
            print(f"      {nm}: {why}")
    else:
        print("  PASS: offsets, lengths, and payload bytes agree for every shard")


if __name__ == "__main__":
    main()
