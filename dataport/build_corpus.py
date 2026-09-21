"""Convert raw Arrow and Parquet corpora to a contiguous mmap layout.

Each series is stored contiguously in a float32 values array.  A parallel
offset array provides O(1) row access without per-series file opens or seeks.
Lengths and data order are preserved; training chooses context limits.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np


def _ts_epoch(t) -> int:
    """Convert a timestamp-like value to epoch seconds."""
    if isinstance(t, (list, tuple)):
        t = t[0] if t else 0
    try:
        if hasattr(t, "timestamp"):
            return int(t.timestamp())
        if isinstance(t, (int, float)):
            return int(t)
        return int(np.datetime64(str(t), "s").astype("int64"))
    except Exception:
        return 0


def iter_arrow_series(path: str):
    """Yield ``(item_id, frequency, values, start_time)`` rows from Arrow IPC.

    The expected schema uses a target column, an optional start or timestamp
    column, and optional frequency and item identifier columns.
    """
    from pyarrow import ipc
    with open(path, "rb") as fh:
        try:
            reader = ipc.open_stream(fh)
            table = reader.read_all()
        except Exception:
            fh.seek(0)
            reader = ipc.open_file(fh)
            table = reader.read_all()
    cols = table.column_names
    freq = table.column("freq").to_pylist() if "freq" in cols else ["?"] * table.num_rows
    iid = table.column("item_id").to_pylist() if "item_id" in cols else [str(i) for i in range(table.num_rows)]
    # Preserve the start timestamp when available; otherwise emit zero.
    if "start" in cols:
        st_col = table.column("start").to_pylist()
    elif "timestamp" in cols:
        st_col = table.column("timestamp").to_pylist()
    else:
        st_col = [0] * table.num_rows
        print(
            f"    [warn] {path.split('/')[-1]}: no start/timestamp column; using 0",
            flush=True,
        )
    if "target" not in cols:
        return
    col = table.column("target")
    # Use Arrow offset and value buffers directly for list-valued targets.
    fast = _flat_series_view(col.combine_chunks())
    if fast is not None and len(fast[1]) == col.length() * fast[2] + 1:
        flat, offs, n_chan = fast
        valid = _valid_mask(col) if col.null_count else None
        for i in range(col.length()):
            if valid is not None and not valid[i]:
                continue
            ts0 = _ts_epoch(st_col[i])
            f = str(freq[i])
            if n_chan == 1:
                a = _as_f32(flat[offs[i]:offs[i + 1]])
                # Keep NaNs; validity is derived by the data adapter.
                if a.size >= 16:
                    yield str(iid[i]), f, a, ts0
            else:
                # Emit each channel as a separate univariate series.
                for c in range(n_chan):
                    j = i * n_chan + c
                    row = _as_f32(flat[offs[j]:offs[j + 1]])
                    if row.size >= 16:
                        yield f"{iid[i]}#c{c}", f, row, ts0
        return

    # Fallback for nested or otherwise non-canonical column types.
    tgt = col.to_pylist()
    for t, f, i, st in zip(tgt, freq, iid, st_col):
        if t is None:
            continue
        a = np.asarray(t, dtype=np.float32)
        if a.ndim == 2:
            for c in range(a.shape[0]):
                row = a[c]
                # Keep NaNs; validity is derived by the data adapter.
                if row.size >= 16:
                    yield f"{i}#c{c}", str(f), row, _ts_epoch(st)
        else:
            # Keep NaNs; validity is derived by the data adapter.
            if a.size < 16:
                continue
            yield str(i), str(f), a, _ts_epoch(st)


def infer_freq(ts_s: np.ndarray) -> str:
    """Infer a frequency label from the median timestamp step."""
    return infer_freq_int_ts(ts_s, 1.0)


_FREQ_TAB = [(1, "S"), (60, "T"), (600, "10T"), (900, "15T"), (1800, "30T"), (3600, "H"),
             (21600, "6H"), (86400, "D"), (604800, "W"), (2592000, "M"),
             (7776000, "Q-DEC"), (31536000, "A-DEC")]


def _freq_from_median(med: float) -> str:
    """Map a median step in seconds to a frequency label."""
    if not np.isfinite(med) or med <= 0:
        return "?"
    best = min(_FREQ_TAB, key=lambda kv: abs(np.log(med) - np.log(kv[0])))
    return best[1] if abs(np.log(med) - np.log(best[0])) < 0.2 else "?"


def infer_freq_int_ts(vals_int: np.ndarray, div: float, max_diffs: int = 200_000) -> str:
    """Infer a frequency label from integer timestamps and a unit divisor."""
    if vals_int.size < 3:
        return "?"
    # Subsample differences, not timestamps: sampling first would double the step.
    d = np.diff(vals_int.astype(np.float64, copy=False))
    d = d[d > 0]
    if d.size == 0:
        return "?"
    if d.size > max_diffs:
        step = int(np.ceil(d.size / max_diffs))
        d = d[::step]
    return _freq_from_median(float(np.median(d)) / div)


def _as_f32(a: np.ndarray) -> np.ndarray:
    """Return a float32 view or copy while preserving NaNs."""
    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)


def _valid_mask(col):
    """Return an Arrow column validity mask as a NumPy boolean array."""
    import pyarrow.compute as pc
    return pc.is_valid(col).to_numpy(zero_copy_only=False)


def _flat_series_view(col):
    """Return a zero-copy ``(flat, offsets, channels)`` view when supported."""
    import pyarrow as pa
    typ = col.type
    n_chan = 1
    if pa.types.is_fixed_size_list(typ):
        inner_t = typ.value_type
        if not (pa.types.is_list(inner_t) or pa.types.is_large_list(inner_t)):
            return None
        n_chan = int(typ.list_size)
        inner = col.flatten()
    elif pa.types.is_list(typ) or pa.types.is_large_list(typ):
        inner = col
    else:
        return None
    if not (pa.types.is_list(inner.type) or pa.types.is_large_list(inner.type)):
        return None
    vt = inner.type.value_type
    if not (pa.types.is_integer(vt) or pa.types.is_floating(vt)):
        return None
    vals = inner.values
    if not (pa.types.is_integer(vals.type) or pa.types.is_floating(vals.type)):
        return None
    offs = inner.offsets.to_numpy(zero_copy_only=False)
    flat = vals.to_numpy(zero_copy_only=False)
    if offs.size == 0 or flat.size == 0:
        return None
    return flat, offs, n_chan


def _ts_list_view(col):
    """Return an integer timestamp view for a list-valued timestamp column."""
    import pyarrow as pa
    typ = col.type
    if not (pa.types.is_list(typ) or pa.types.is_large_list(typ)):
        return None
    vt = typ.value_type
    if not pa.types.is_timestamp(vt):
        return None
    div = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(vt.unit)
    if div is None or vt.tz is not None:
        return None
    npv = col.values.to_numpy(zero_copy_only=False)
    if getattr(npv.dtype, "kind", "") == "M":
        ints = npv.view("int64")
        div = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[np.datetime_data(npv.dtype)[0]]
    else:
        ints = col.values.cast(pa.int64()).to_numpy(zero_copy_only=False)
    if ints.size == 0:
        return None
    return ints, col.offsets.to_numpy(zero_copy_only=False), div


def _emit_column(t, val_c: str, id_c, ts_c, path: str):
    """Emit one value column in either array-column or long-table form."""
    import pyarrow.parquet as pq  # noqa: F401 - keep the dependency explicit

    multi = val_c != "target"
    tag = "" if not multi else f"@{val_c}"
    col = t.column(val_c)
    # Keep list-valued columns on the zero-copy path.
    fast = _flat_series_view(col.combine_chunks())
    if fast is not None and fast[2] == 1 and len(fast[1]) == col.length() + 1:
        yield from _emit_list_fast(t, col.length(), fast[0], fast[1],
                                  id_c, ts_c, path, tag)
        return
    vals = col.to_pylist()
    tss = t.column(ts_c).to_pylist() if ts_c else [None] * len(vals)
    ids = t.column(id_c).to_pylist() if id_c else [str(i) for i in range(len(vals))]
    n_out = 0
    arr_like = bool(vals) and isinstance(vals[0], (list, tuple))
    if arr_like:
        for i, (v, ts, sid) in enumerate(zip(vals, tss, ids)):
            if v is None:
                continue
            a = np.asarray(v, dtype=np.float32).ravel()
            if np.isfinite(a).sum() < 16:
                continue
            freq = "?"
            if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                try:
                    arr = np.asarray([x.timestamp() if hasattr(x, "timestamp") else float(x)
                                      for x in ts], dtype=np.float64)
                    freq = infer_freq(arr)
                except Exception:
                    freq = "?"
            n_out += 1
            yield f"{sid}{tag}#r{i}", freq, a, (int(ts[0].timestamp()) if hasattr(ts[0], "timestamp") else (int(ts[0]) if ts else 0))
    else:
        groups: dict[str, list[tuple[float, float]]] = {}
        for sid, tt, vv in zip(ids, tss, vals):
            if vv is None or tt is None:
                continue
            try:
                tv = tt.timestamp() if hasattr(tt, "timestamp") else float(tt)
            except Exception:
                continue
            groups.setdefault(str(sid), []).append((tv, float(vv)))
        for sid, rows in groups.items():
            rows.sort(key=lambda x: x[0])
            ts_arr = np.asarray([r[0] for r in rows], dtype=np.float64)
            v_arr = np.asarray([r[1] for r in rows], dtype=np.float32)
            # Keep NaNs; validity is derived by the data adapter.
            if v_arr.size >= 16:
                n_out += 1
                yield f"{sid}{tag}", infer_freq(ts_arr), v_arr, int(ts_arr[0])
    if n_out == 0:
        print(
            f"  [warn] {os.path.basename(os.path.dirname(path))}/"
            f"{os.path.basename(path)} column {val_c}: produced no series",
            flush=True,
        )


def _emit_list_fast(t, n_rows: int, flat: np.ndarray, offs: np.ndarray,
                    id_c, ts_c, path: str, tag: str):
    """Emit a list-valued column with one series per row."""
    ids = t.column(id_c).to_pylist() if id_c else [str(i) for i in range(n_rows)]
    ts_col = t.column(ts_c).combine_chunks() if ts_c else None
    ts_view = _ts_list_view(ts_col) if ts_col is not None else None
    n_out = 0
    for i in range(n_rows):
        a = _as_f32(flat[offs[i]:offs[i + 1]])
        # Keep NaNs and drop rows with fewer than 16 finite observations.
        if a.size == 0 or int(np.isfinite(a).sum()) < 16:
            continue
        freq, ts0 = "?", 0
        if ts_col is not None:
            # Convert through Python datetime to match datetime.timestamp semantics.
            try:
                ts0 = _ts_epoch(ts_col[i][0].as_py())
            except Exception:
                ts0 = 0
        if ts_view is not None:
            tv, toff, div = ts_view
            seg = tv[toff[i]:toff[i + 1]]
            if seg.size >= 3:
                # Frequency uses step sizes, so a constant timezone offset is harmless.
                freq = infer_freq_int_ts(seg, div)
        n_out += 1
        yield f"{ids[i]}{tag}#r{i}", freq, a, ts0
    if n_out == 0:
        print(
            f"  [warn] {os.path.basename(os.path.dirname(path))}/"
            f"{os.path.basename(path)} column {tag or 'target'}: produced no series",
            flush=True,
        )


def iter_parquet_generic(path: str):
    """Read Parquet corpora with heterogeneous schemas.

    Target or numeric arrays are emitted as per-row series.  Scalar tables are
    grouped by identifier and sorted by timestamp.  Missing timestamps leave the
    frequency unknown rather than dropping the series.
    """
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = t.column_names
    id_c = next((c for c in ("id", "item_id", "series_id") if c in cols), None)
    ts_c = next((c for c in ("timestamp", "start", "time", "date", "ds") if c in cols), None)
    # Select numeric value columns explicitly and avoid forecast covariates.
    import pyarrow as pa
    skip = {id_c, ts_c, "latitude", "longitude", "level", "subset", "freq", "item_id", "start"}
    meta_prefixes = ("fcst", "forecast", "pred")

    def _is_numeric_like(field) -> bool:
        t = field.type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            return True
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            return _is_numeric_like(pa.field("x", t.value_type))
        return False

    if "target" in cols:
        val_cols = ["target"]
    else:
        val_cols = [c for c in cols
                    if c not in skip
                    and not str(c).lower().startswith(meta_prefixes)
                    and _is_numeric_like(t.schema.field(c))]
    if not val_cols:
        print(
            f"  [warn] {os.path.basename(os.path.dirname(path))}/"
            f"{os.path.basename(path)}: no numeric columns found "
            f"(columns={cols}); skipping",
            flush=True,
        )
        return
    t = t.select([c for c in (id_c, ts_c, *val_cols) if c and c in cols])
    val_c = val_cols[0]
    # Emit every numeric column as one or more independent series.
    for _vc in val_cols:
        yield from _emit_column(t, _vc, id_c, ts_c, path)



def build_corpus(src: str, out: str, shard_gb: float = 8.0, max_files: int = 0,
                 dry: bool = False, part: int = 0, parts: int = 1,
                 balance: str = "index", file_mod: int = 0,
                 file_mod_n: int = 1) -> dict:
    """Convert Arrow and Parquet corpora to the fast format.

    Parameters:
        src: source root whose first-level directories are dataset names.
        out: output directory.
        shard_gb: target shard size in GiB.
        max_files: maximum files per dataset; zero means unlimited.
        dry: count records without writing arrays.
        part, parts: deterministic partition of datasets across processes.
        balance: ``index`` for round-robin or ``bytes`` for LPT load balance.
        file_mod, file_mod_n: optional file-level split within each dataset.
    """

    os.makedirs(out, exist_ok=True)
    # Ignore hidden directories and directories without data files.
    all_ds_raw = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    all_ds = []
    for _d in all_ds_raw:
        if str(_d).startswith("."):
            continue
        _p = os.path.join(src, _d)
        _has = False
        for _root, _dirs, _files in os.walk(_p):
            if any(f.endswith((".arrow", ".parquet")) for f in _files):
                _has = True
                break
        if _has:
            all_ds.append(_d)
        else:
            print(f"  [skip] {_d}: no Arrow or Parquet files", flush=True)

    if balance == "bytes" and parts > 1:
        # Longest-processing-time first assignment balances large datasets.
        size = {}
        for d in all_ds:
            tot = 0
            for root, _dirs, files in os.walk(os.path.join(src, d)):
                for fn in files:
                    try:
                        tot += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            size[d] = tot
        buckets: list[list[str]] = [[] for _ in range(parts)]
        load = [0] * parts
        for d in sorted(all_ds, key=lambda x: -size[x]):
            k = load.index(min(load))
            buckets[k].append(d)
            load[k] += size[d]
        print(
            "  [balance=bytes] estimated part loads: "
            + ", ".join(f"p{i}={v / 1e9:.1f}G" for i, v in enumerate(load)),
            flush=True,
        )
        datasets = buckets[part]
    elif parts > 1:
        datasets = [d for i, d in enumerate(all_ds) if i % parts == part]
    else:
        datasets = all_ds
    print(
        f"  datasets {len(datasets)}/{len(all_ds)} (part {part}/{parts}); "
        f"target shard size {shard_gb} GB"
    )

    limit = int(shard_gb * (1024 ** 3) / 4)
    shard_idx = 0
    buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
    meta_rows = []
    ds_ids = {d: i for i, d in enumerate(datasets)}
    # Persist the local dataset identifier mapping for this part.
    with open(f"{out}/ds_names.json", "w") as _fh:
        json.dump({"part": part, "parts": parts, "src": src,
                   "datasets": {str(i): d for i, d in enumerate(datasets)}},
                  _fh, ensure_ascii=False, indent=1)
    freq_ids: dict[str, int] = {}
    written_pts = 0
    written_rows = 0
    cur_pts = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, buf_rows, buf_off, buf_len, buf_freq, buf_ds, written_pts, written_rows, cur_pts, buf_ts
        if not buf_len:
            return
        # Filter short series and rebuild offsets so all arrays stay aligned.
        _fin = [int(np.isfinite(a).sum()) for a in buf_rows]
        if any(f < 12 for f in _fin):
            keep = [i for i, f in enumerate(_fin) if f >= 12]
            dropped = len(buf_len) - len(keep)
            if dropped:
                print(
                    f"    [length guard] dropped {dropped} series with fewer than 12 finite values",
                    flush=True,
                )
            if not keep:
                buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
                cur_pts = 0
                return
            buf_rows = [buf_rows[i] for i in keep]
            buf_len = [buf_len[i] for i in keep]
            buf_freq = [buf_freq[i] for i in keep]
            buf_ds = [buf_ds[i] for i in keep]
            # Keep timestamps aligned with the retained rows.
            buf_ts = [buf_ts[i] for i in keep]
            buf_off = [0]
            for _L in buf_len:
                buf_off.append(buf_off[-1] + _L)
        vals = np.concatenate(buf_rows).astype(np.float32, copy=False)
        np.save(f"{out}/shard{shard_idx:04d}.values.f32.npy", vals)
        np.save(f"{out}/shard{shard_idx:04d}.offsets.npy", np.asarray(buf_off, dtype=np.int64))
        np.save(f"{out}/shard{shard_idx:04d}.lengths.npy", np.asarray(buf_len, dtype=np.int32))
        np.save(f"{out}/shard{shard_idx:04d}.freq_id.npy", np.asarray(buf_freq, dtype=np.int16))
        np.save(f"{out}/shard{shard_idx:04d}.ds_id.npy", np.asarray(buf_ds, dtype=np.int16))
        np.save(f"{out}/shard{shard_idx:04d}.ts.npy", np.asarray(buf_ts, dtype=np.int64))
        written_pts += int(vals.size)
        written_rows += len(buf_len)
        print(
            f"    shard{shard_idx:04d}: {len(buf_len):>7d} series / "
            f"{vals.size / 1e9:5.2f}B points / {vals.nbytes / 1e9:5.2f} GB "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        meta_rows.append((shard_idx, len(buf_len), int(vals.size)))
        shard_idx += 1
        buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
        cur_pts = 0

    n_files = 0
    for ds in datasets:
        # Dispatch by file format: Arrow for wide tables, Parquet otherwise.
        fs_arrow = sorted(glob.glob(os.path.join(src, ds, "**", "*.arrow"), recursive=True))
        fs_parq = sorted(glob.glob(os.path.join(src, ds, "**", "*.parquet"), recursive=True))
        # Exclude Hugging Face cache duplicates from the source listing.
        _before = len(fs_arrow) + len(fs_parq)
        fs_arrow = [f for f in fs_arrow if not os.path.basename(f).startswith("cache-")]
        fs_parq = [f for f in fs_parq if not os.path.basename(f).startswith("cache-")]
        _dropped = _before - len(fs_arrow) - len(fs_parq)
        if _dropped:
            print(
                f"    [deduplicate] {ds}: excluded {_dropped} cache copies",
                flush=True,
            )
        if fs_parq and not fs_arrow:
            files = fs_parq
            reader_fn = iter_parquet_generic
        else:
            files = fs_arrow
            reader_fn = iter_arrow_series
        if file_mod_n > 1:
            # Optional file-level split within one dataset.
            _n_all = len(files)
            files = [f for i, f in enumerate(files) if i % file_mod_n == file_mod]
            print(
                f"    [file_mod] {ds}: shard {file_mod}/{file_mod_n} -> "
                f"{len(files)}/{_n_all} files",
                flush=True,
            )
        for fp in files:
            if max_files and n_files >= max_files:
                break
            n_files += 1
            for _iid, freq, arr, _ts0 in reader_fn(fp):
                if not dry:
                    buf_rows.append(arr)
                    buf_off.append(buf_off[-1] + arr.size)
                    buf_len.append(arr.size)
                    if freq not in freq_ids:
                        freq_ids[freq] = len(freq_ids)
                    buf_freq.append(freq_ids[freq])
                    buf_ts.append(int(_ts0))
                    buf_ds.append(ds_ids[ds])
                    cur_pts += arr.size
                else:
                    written_pts += arr.size
                    written_rows += 1
                # Flush regularly to bound memory and expose progress.
                if len(buf_off) - 1 >= 50_000 or cur_pts >= limit:
                    if not dry:
                        flush()
            if max_files and n_files >= max_files:
                break
        if n_files % 50 == 0:
            print(
                f"    ... processed {n_files} files / {written_rows:,} series",
                flush=True,
            )
    if not dry:
        flush()

    # Write global metadata and length buckets without reordering data.
    if not dry:
        with open(f"{out}/datasets.txt", "w") as fh:
            fh.write("\n".join(datasets) + "\n")
        with open(f"{out}/freqs.txt", "w") as fh:
            fh.write("\n".join(sorted(freq_ids, key=lambda k: freq_ids[k])) + "\n")
        lengths_all, freq_all, ds_all = [], [], []
        for k in range(shard_idx):
            lengths_all.append(np.load(f"{out}/shard{k:04d}.lengths.npy"))
            freq_all.append(np.load(f"{out}/shard{k:04d}.freq_id.npy"))
            ds_all.append(np.load(f"{out}/shard{k:04d}.ds_id.npy"))
        L = np.concatenate(lengths_all) if lengths_all else np.zeros(0, np.int32)
        F = np.concatenate(freq_all) if freq_all else np.zeros(0, np.int16)
        D = np.concatenate(ds_all) if ds_all else np.zeros(0, np.int16)
        edges = np.array([64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 30])
        buckets = {f"le{int(e)}": np.where(L <= e)[0].astype(np.int64) for e in edges}
        np.savez(f"{out}/index.npz", lengths=L, freq_id=F, ds_id=D, **buckets)
        manifest = {"n_shards": shard_idx, "n_rows": int(L.size),
                    "n_points": int(L.sum()), "shards": meta_rows,
                    "n_datasets": len(datasets),
                    "src": src, "part": part, "parts": parts,
                    "file_mod": [file_mod, file_mod_n],
                    "datasets": datasets,
                    "notes": "contiguous float32 values; O(1) offsets; no length truncation"}
        with open(f"{out}/manifest.json", "w") as _fh:
            json.dump(manifest, _fh, indent=1)
        print(
            f"  complete: {shard_idx} shards / {L.size:,} series / "
            f"{L.sum() / 1e9:.2f}B points / {L.sum() * 4 / 1e9:.1f} GB "
            f"[{time.time() - t0:.0f}s]"
        )
        summary = {"out": out, "n_shards": shard_idx, "n_rows": int(L.size),
                   "n_points": int(L.sum()), "n_datasets": len(datasets),
                   "dry": False}
    else:
        print(
            f"  [dry] would write {written_rows:,} series / "
            f"{written_pts / 1e9:.2f}B points / {written_pts * 4 / 1e9:.1f} GB"
        )
        summary = {"out": out, "n_shards": 0, "n_rows": int(written_rows),
                   "n_points": int(written_pts), "n_datasets": len(datasets),
                   "dry": True}
    return summary
