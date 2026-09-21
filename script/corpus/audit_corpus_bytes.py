"""Independently audit the fast corpus without trusting its manifest.

The first pass compares official remote manifests with local files. This pass
reads only generated corpus files and recomputes totals three ways:

- byte method: value-array payload bytes divided by four float32 points;
- length method: the sum of every shard lengths array;
- offset method: each final offset and the total number of sequences.

All three must agree. The manifest is then checked against those recomputed
values rather than treated as evidence.

Optional spot checks validate row lengths and require at least 12 finite
points. NaNs are preserved intentionally; masks are derived downstream.

Usage:
    env -u PYTHONPATH .venv/bin/python script/corpus/audit_corpus_bytes.py \
        --corpus data/corpus_fast
    ... --src data/pretrain_full --spot 20
"""


from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np


def npy_data_bytes(path: str) -> int:
    """Return the payload byte count of an .npy file, excluding magic and header."""

    # np.save writes a header, so raw file size divided by four overcounts points.
    # Independent recomputation catches this kind of layout-specific error.

    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic[:6] != b"\x93NUMPY":
            return os.path.getsize(path)
        hlen = int(np.frombuffer(fh.read(2), dtype=np.uint16)[0])
        return os.path.getsize(path) - 10 - hlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--spot", type=int, default=0, help="Number of value spot checks")
    ap.add_argument("--src", default=None, help="Optional original Arrow root")
    a = ap.parse_args()

    # Support both partitioned output and the older flat layout.
    shards = sorted(glob.glob(f"{a.corpus}/shard*.values.f32.npy"))
    if not shards:
        shards = sorted(glob.glob(f"{a.corpus}/p*/shard*.values.f32.npy"))
    print(f"  value shards: {len(shards)}")
    if not shards:
        print("  FAIL: no value files found"); sys.exit(2)

    # Method 1: payload bytes.
    total_bytes = sum(npy_data_bytes(f) for f in shards)
    m1 = total_bytes // 4
    print(f"  M1 bytes: {total_bytes:,} payload bytes -> {m1:,} points")

    # Method 2: lengths; Method 3: offsets.
    m2 = 0
    m3_rows = 0
    bad = []
    for vf in shards:
        base = vf[:-len(".values.f32.npy")]
        lf, of = base + ".lengths.npy", base + ".offsets.npy"
        if not (os.path.exists(lf) and os.path.exists(of)):
            bad.append((os.path.basename(base), "missing lengths/offsets"))
            continue
        L = np.load(lf)
        O = np.load(of)
        m2 += int(L.sum())
        m3_rows += len(O) - 1
        if len(O) - 1 != len(L):
            bad.append((os.path.basename(base), f"row mismatch offsets={len(O)-1} lengths={len(L)}"))
        elif int(O[-1]) != int(L.sum()):
            bad.append((os.path.basename(base), f"final offset {int(O[-1])} != length sum {int(L.sum())}"))
        elif int(O[-1]) * 4 != npy_data_bytes(vf):
            bad.append((os.path.basename(base),
                        f"final offset bytes={int(O[-1])*4} != file bytes={os.path.getsize(vf)}"))
    print(f"  M2 lengths: {m2:,} points")
    print(f"  M3 offsets: {m3_rows:,} sequences with per-shard checks")

    ok = (m1 == m2)
    print(f"\n  {'PASS' if ok else 'FAIL'} method agreement: bytes={m1:,}, lengths={m2:,}"
          f"{'' if ok else f', delta={m1-m2:,}'}")
    if bad:
        print(f"  FAIL: {len(bad)} shard-level problems")
        for nm, why in bad[:10]:
            print(f"      {nm}: {why}")
    else:
        print("  PASS: offsets, lengths, and payload bytes agree for every shard")

    # Compare with the manifest as a claim, not as a source of truth.
    mf = f"{a.corpus}/manifest.json"
    if os.path.exists(mf):
        man = json.load(open(mf))
        d_rows = m3_rows - int(man.get("n_rows", -1))
        d_pts = m2 - int(man.get("n_points", -1))
        print(f"  manifest claims {man.get('n_rows'):,} rows / {man.get('n_points'):,} points")
        print(f"  {'PASS' if d_rows == 0 and d_pts == 0 else 'FAIL'} manifest delta: "
              f"rows={d_rows:+,}, points={d_pts:+,}")

    # Spot-check row lengths and finite-point invariants.
    if a.spot:
        rs = random.Random(0)
        idx = [rs.randrange(len(shards)) for _ in range(a.spot)]
        n_ok = n_bad = 0
        for si in idx:
            vf = shards[si]
            base = vf[:-len(".values.f32.npy")]
            V = np.load(vf, mmap_mode="r")
            L = np.load(base + ".lengths.npy")
            O = np.load(base + ".offsets.npy")
            i = rs.randrange(len(L))
            row = np.asarray(V[O[i]:O[i + 1]])
            # NaN is preserved; downstream derives masks. Require length agreement
            # and at least 12 finite points, matching the writer invariant.
            good = (row.size == int(L[i])) and int(np.isfinite(row).sum()) >= 12
            n_ok += good
            n_bad += (not good)
            if good and a.src:
                pass  # Independent Arrow comparison is performed separately.
        print(f"\n  value checks: {n_ok} passed / {n_bad} failed")

    print("\nThe corpus is self-consistent when M1 equals M2, shards validate,")
    print("and manifest deltas are zero. Combine this with source-manifest auditing.")


if __name__ == "__main__":
    main()
