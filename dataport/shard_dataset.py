"""Training dataset for compact shards with evaluation-aligned windows.

The sample construction mirrors the evaluation predictor: contexts are built
from the same scale levels, timestamps, normalization statistics, coverage
masks, and raw-resolution forecast targets.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from dataport.prefetch import ShardPrefetcher
from module.pyramid.levels import level_span, resolve_level_widths
from util.freq import freq_to_seconds, get_seasonality


def revin_level_stats(x: np.ndarray, xm: np.ndarray, eps: float = 1e-5):
    """Compute per-level RevIN statistics with a biased variance estimate."""
    out = []
    for li in range(x.shape[0]):
        m = xm[li].astype(bool)
        if m.any():
            v = x[li][m].astype(np.float64)
            mu = float(v.mean())
            sd = float(np.sqrt(((v - mu) ** 2).mean() + eps))
        else:
            mu, sd = 0.0, 1.0
        out.append((mu, sd))
    return out


def robust_stats(flat_x: np.ndarray, flat_m: np.ndarray, mode: str,
                 max_abs: float = 20.0) -> tuple[float, float]:
    """Compute the NumPy equivalent of the model's robust normalization fit."""
    m = flat_m.astype(bool)
    if not m.any():
        return 0.0, 1.0
    # Replace invalid entries before computing sums to avoid propagating infinities.
    x_fin = np.where(m, flat_x, 0.0)
    m = m & np.isfinite(flat_x)
    if not m.any():
        return 0.0, 1.0
    vals = np.sort(flat_x[m])
    cnt = vals.size
    n = flat_x.size
    if mode == "robust":
        def pick(q: float) -> float:
            return float(vals[min(int(cnt * q), n - 1)])
        loc = pick(0.5)
        scale = pick(0.75) - pick(0.25)
    elif mode == "arcsinh":
        # Match the official causal window statistics: valid-point mean and std.
        mean = float(flat_x[m].mean())
        scale = float(flat_x[m].std())
        loc = mean
    else:  # standard
        mean = float(flat_x[m].mean())
        scale = float(flat_x[m].std())
        loc = mean
    rms = float(np.sqrt((x_fin ** 2 * m).sum() / max(m.sum(), 1.0)))
    floor = max(1e-2, 0.05 * rms)
    if not np.isfinite(scale) or scale <= 0:
        scale = rms
    scale = max(scale, floor)
    if not np.isfinite(loc):
        loc = 0.0
    return loc, scale


class ShardDataset(data.Dataset):
    """Dataset over compact shards with memmap access and aligned windows.

    Training samples random windows; validation samples use a fixed cut point.
    """

    def __init__(self, shard_dir, cfg, train=True, cache_shards: int = 64):
        self.dir = Path(shard_dir)
        compact_path = self.dir / "index_compact.npz"
        if not compact_path.exists():
            raise FileNotFoundError(
                f"missing compact index {compact_path}; run script/tools/make_compact_index.py")
        with np.load(compact_path, allow_pickle=False) as z:
            shard_files = z["shard_files"].tolist()
            freqs = z["freqs"].tolist()
            shard_id = z["shard_id"]
            row_id = z["row_id"]
            freq_id = z["freq_id"]

        self._shard_files = shard_files
        self._freqs = freqs
        keep = self._split_mask(len(shard_id), cfg, train)
        self._shard_ids = shard_id[keep]
        self._row_ids = row_id[keep]
        self._freq_ids = freq_id[keep]

        self._init_windows(cfg, train, cache_shards)
        # Prefer the memmap layout and fall back to compressed NPZ shards.
        probe = self.dir / (self._stem(shard_files[0]) + ".values.npy")
        self.fast_layout = probe.exists()
        self._ts_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _split_mask(n: int, cfg, train: bool) -> np.ndarray:
        """Return a deterministic train/validation split mask."""
        frac = cfg["data"]["val_fraction"]
        pos = np.arange(n) % 10
        cut = round((1 - frac) * 10)
        return pos < cut if train else pos >= cut

    def _init_windows(self, cfg, train: bool, cache_shards: int) -> None:
        """Initialize shared window, normalization, and augmentation state."""
        self.cfg = cfg
        self.train = train
        self.H = int(cfg["model"]["horizon"])
        self.ratios = list(cfg["pyramid"]["ratios"])
        self.L = len(self.ratios) + 1
        # Token widths are ordered from coarsest to finest levels.
        self.level_widths = resolve_level_widths(
            cfg["model"].get("level_widths", cfg["model"]["W"]), self.L)
        self.W = self.level_widths[-1]
        self.W_out = max(self.level_widths)
        # A non-positive cap disables truncation.
        _cap = int((cfg.get("data") or {}).get("ctx_len_cap", 4096) or 0)
        _span = level_span(self.level_widths, self.ratios)
        self.ctx_len = _span if _cap <= 0 else min(_span, _cap)
        self.norm_mode = cfg["model"].get("context_norm", "robust")
        # ``revin`` normalizes per pyramid level; ``robust_global`` uses one window.
        self.level_norm = cfg["data"].get("level_norm", "robust_global")
        # Optional seasonal-copy reference used by the committing objective.
        self.commit_w = float((cfg.get("train") or {}).get("commit_w", 0.0) or 0.0)
        self._sf_cache: dict[str, float] = {}
        self._sf_missing: set[str] = set()
        # Cache frequency-derived quantities and the relative time grid.
        self._s_cache: dict[str, int] = {}
        self._sec_cache: dict[str, float] = {}
        # Build per-level relative time axes padded to the common width.
        self._xt_full = np.stack([
            np.pad(np.arange(w, dtype=np.float32) / np.float32(w),
                   (self.W_out - w, 0)) for w in self.level_widths])
        self.hole_prob = float(cfg["data"].get("hole_prob", 0.0)) if train else 0.0
        # Raw-window mode passes the native-resolution window to the model.
        self.window_mode = str((cfg.get("data") or {}).get("window_mode", "levels"))
        self.ar_chunks = int((cfg.get("train") or {}).get("ar_chunks", 1) or 1)
        self.emit_window = bool(train and self.window_mode == "raw")
        self.tgt_span = self.H * (self.ar_chunks if self.emit_window else 1)
        self.win_len = self.ctx_len + self.tgt_span
        # Optional per-row seasonal scale override.
        self._row_scales = None
        self.seed = int(cfg["data"].get("seed", 42)) + (0 if train else 1)
        self._cache_shards = cache_shards
        self._cache: OrderedDict[str, dict] = OrderedDict()

    # Layout-independent row access.
    def _sample_rng(self, i: int) -> np.random.Generator:
        """Return a row-local RNG independent of worker scheduling and resume order."""
        return np.random.default_rng((int(self.seed), int(i)))

    def _stem(self, fname: str) -> str:
        return fname[:-4] if fname.endswith(".npz") else fname

    def _load_row(self, shard_file: str, row: int
                  ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(values, valid, start_time)`` for one row."""
        if self.fast_layout:
            stem = self._stem(shard_file)
            if stem not in self._cache:
                self._cache[stem] = {
                    "values": np.load(self.dir / f"{stem}.values.npy",
                                      mmap_mode="r"),
                    "valid": np.load(self.dir / f"{stem}.valid.npy",
                                     mmap_mode="r"),
                }
                self._cache.move_to_end(stem)
                while len(self._cache) > self._cache_shards:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(stem)
            item = self._cache[stem]
            if stem not in self._ts_cache:
                self._ts_cache[stem] = np.load(self.dir / f"{stem}.ts.npy")
                while len(self._ts_cache) > self._cache_shards:
                    self._ts_cache.popitem(last=False)
            ts0 = int(self._ts_cache[stem][row])
            return (np.asarray(item["values"][row], dtype=np.float32),
                    np.asarray(item["valid"][row], dtype=bool), ts0)

        if shard_file not in self._cache:
            with np.load(self.dir / shard_file, allow_pickle=False) as z:
                self._cache[shard_file] = {
                    "values": np.ascontiguousarray(z["values"]),
                    "valid": np.ascontiguousarray(z["valid"]),
                    "ts": np.asarray(z["ts"]),
                }
            self._cache.move_to_end(shard_file)
            while len(self._cache) > self._cache_shards:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(shard_file)
        item = self._cache[shard_file]
        return (item["values"][row, :, 0].astype(np.float32),
                item["valid"][row, :, 0].astype(bool),
                int(item["ts"][row]))

    def _build_levels(self, values: np.ndarray, valid: np.ndarray):
        """Build pyramid levels bottom-up, with level zero at native resolution."""
        levels = [(values, valid, valid.astype(np.float32))]
        for r in self.ratios:
            pv, pm, _ = levels[-1]
            T = (pv.shape[0] // r) * r
            if T < r:
                break
            v2 = pv[:T].reshape(T // r, r)
            m2 = pm[:T].reshape(T // r, r)
            cnt = m2.sum(axis=1)
            # Zero invalid entries before aggregation so NaNs cannot propagate.
            pv0 = np.where(m2, v2, 0.0)
            agg = np.where(cnt > 0, pv0.sum(axis=1) / np.maximum(cnt, 1), 0.0)
            levels.append((agg.astype(np.float32), cnt > 0, cnt / float(r)))
        # Duplicate the finest level at the front when fewer levels are available.
        while len(levels) < self.L:
            levels.insert(0, levels[0])
        return levels

    def _seasonal_scale(self, freq: str):
        """Cached wrapper around the upstream seasonal scale factor."""
        if freq in self._sf_cache:
            return self._sf_cache[freq]
        if freq in self._sf_missing:
            return None
        try:
            from module.losses.tinycast import seasonal_scale_factor
            sf = float(seasonal_scale_factor(freq, None))
        except Exception:                                            # noqa: BLE001
            self._sf_missing.add(freq)
            return None
        self._sf_cache[freq] = sf
        return sf

    def _seasonal_copy(self, ctx_v, ctx_m, loc: float, scale: float, freq: str):
        """Return the seasonal-copy reference in normalized target space."""
        import torch as _t
        from module.losses.tinycast import seasonal_copy_baseline
        sf = self._seasonal_scale(freq)
        if sf is None:
            return None
        v = np.asarray(ctx_v, dtype=np.float32)
        m = np.asarray(ctx_m, dtype=bool)
        if v.size < 4 or not m.any():
            return None
        if not m.all():
            idx = np.where(m, np.arange(m.size), 0)
            np.maximum.accumulate(idx, out=idx)
            v = v[idx]
            first = int(np.argmax(m))
            if first > 0:
                v[:first] = v[first]
        x = (v - float(loc)) / max(float(scale), 1e-12)
        with _t.no_grad():
            c = seasonal_copy_baseline(_t.from_numpy(np.ascontiguousarray(x))[None, :],
                                       int(self.H), float(sf))
        return c[0].numpy().astype(np.float32)

    def _take_tail(self, arr: np.ndarray, width: int) -> np.ndarray:
        """Take the last ``width`` tokens and left-pad to ``self.W_out``."""
        keep = min(int(width), arr.shape[0])
        tail = arr[arr.shape[0] - keep:]
        lead = self.W_out - keep
        if lead <= 0:
            return np.ascontiguousarray(tail)
        pad = np.zeros((lead,) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([pad, tail], axis=0)

    def _pick_t_end(self, values: np.ndarray, valid: np.ndarray,
                    rng: np.random.Generator) -> int:
        """Select the context end position, which is also the forecast start."""
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return max(1, values.shape[0] // 2)
        last = int(idx[-1]) + 1
        if not self.train:
            # Validation uses a fixed cut point for reproducible loss values.
            return max(1, last - self.H) if last > self.H else max(1, last // 2)
        lo = 1
        hi = max(lo + 1, last)
        for _ in range(8):
            t = int(rng.integers(lo, hi))
            cm = valid[max(0, t - self.ctx_len):t]
            # Rollout mode requires observable targets for every chunk.
            tgt = valid[t:t + self.tgt_span]
            if cm.sum() >= 8 and tgt.sum() >= 8:
                return t
        return max(1, last - self.H) if last > self.H else max(1, last // 2)

    def _row_scale(self, row: int, freq: str) -> float:
        """Return the row scale from the sidecar when present, otherwise from frequency."""
        rs = self._row_scales
        if rs is not None and 0 <= row < rs.shape[0]:
            s = float(rs[row])
            if np.isfinite(s) and s > 0:
                return s
        s = self._seasonal_scale(freq)
        return float(s) if s else 1.0

    def _window_sample(self, values, valid, t_end: int, row: int, freq: str):
        """Return a native-resolution window, validity mask, and row scale."""
        start = int(t_end) - self.ctx_len
        pad_l = max(0, -start)
        start = max(0, start)
        end = int(t_end) + self.tgt_span
        v = np.asarray(values[start:end], dtype=np.float32)
        m = np.asarray(valid[start:end], dtype=bool)
        pad_r = max(0, self.win_len - v.shape[0] - pad_l)
        if pad_l or pad_r:
            v = np.concatenate([np.zeros(pad_l, np.float32), v,
                                np.zeros(pad_r, np.float32)])
            m = np.concatenate([np.zeros(pad_l, bool), m,
                                np.zeros(pad_r, bool)])
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return (torch.from_numpy(np.ascontiguousarray(v)),
                torch.from_numpy(np.ascontiguousarray(m)),
                torch.tensor(self._row_scale(row, freq), dtype=torch.float32))
    def _seasonality(self, freq: str) -> int:
        """Return the seasonal period for a frequency, cached by frequency."""
        S = self._s_cache.get(freq)
        if S is None:
            try:
                S = int(get_seasonality(freq))
            except Exception:                                        # noqa: BLE001
                S = 1
            if S <= 0:
                S = 1
            self._s_cache[freq] = S
        return S

    def _sec_per_step(self, freq: str) -> float:
        """Return seconds per step for a frequency, cached by frequency."""
        v = self._sec_cache.get(freq)
        if v is None:
            v = float(freq_to_seconds(freq))
            self._sec_cache[freq] = v
        return v

    def _snaive_anchor(self, ctx: np.ndarray, S: int) -> np.ndarray:
        """Return the seasonal-naive anchor with masked-value fallback.

        Positions repeat the last complete cycle.  Missing values fall back to the
        nearest earlier valid observation and finally to zero.
        """
        H = self.H
        T = ctx.shape[0]
        if S < 1:
            S = 1
        # Repeat the last complete cycle, matching the SeasonalNaive definition.
        anchor = np.empty(H, dtype=np.float32)
        if S <= T:
            pos = (T - S) + (np.arange(H, dtype=np.int64) % S)
            np.take(ctx, pos, out=anchor)
            if not np.isfinite(ctx[T - S:]).all():
                return self._fill_anchor_nan(ctx, anchor, pos)
            return anchor
        # Handle series shorter than one cycle with an explicit deterministic fallback.
        anchor.fill(np.nan)
        for k in range(H):
            i = T - S + (k % S)
            j = i
            while 0 <= j < T and not np.isfinite(ctx[j]):
                j -= 1
            if 0 <= j < T:
                anchor[k] = ctx[j]
            else:
                k2 = k
                while k2 >= 0 and not np.isfinite(anchor[k2]):
                    k2 -= 1
                anchor[k] = anchor[k2] if k2 >= 0 else 0.0
        return anchor

    def _fill_anchor_nan(self, ctx: np.ndarray, anchor: np.ndarray,
                         pos: np.ndarray) -> np.ndarray:
        """Replace invalid anchors with the nearest earlier valid observation."""
        T = ctx.shape[0]
        fin = np.isfinite(ctx)
        if not fin.any():
            anchor.fill(0.0)
            return anchor
        idx = np.where(fin, np.arange(T), -1)
        np.maximum.accumulate(idx, out=idx)
        src = idx[pos]
        ok = src >= 0
        if ok.any():
            anchor[ok] = ctx[src[ok]]
        for k in np.flatnonzero(~ok):
            k2 = int(k) - 1
            while k2 >= 0 and not np.isfinite(anchor[k2]):
                k2 -= 1
            anchor[k] = anchor[k2] if k2 >= 0 else 0.0
        return anchor

    def __len__(self) -> int:
        return max(1, len(self._shard_ids))

    def __getitem__(self, idx: int):
        i = idx % len(self._shard_ids)
        shard_file = self._shard_files[int(self._shard_ids[i])]
        row = int(self._row_ids[i])
        freq = str(self._freqs[int(self._freq_ids[i])])
        values, valid, ts0 = self._load_row(shard_file, row)

        rng = self._sample_rng(i)

        t_end = self._pick_t_end(values, valid, rng)
        # Raw-window mode leaves level construction and normalization to the model.
        if self.emit_window:
            return self._window_sample(values, valid, t_end, int(i), freq)
        t0 = max(0, t_end - self.ctx_len)
        ctx_v = np.asarray(values[t0:t_end], dtype=np.float32)
        ctx_m = np.asarray(valid[t0:t_end], dtype=bool)
        if ctx_v.size == 0:
            ctx_v = np.zeros(1, dtype=np.float32)
            ctx_m = np.zeros(1, dtype=bool)

        # Apply hole augmentation to the training context only.
        if self.train and self.hole_prob > 0 and ctx_m.any():
            hole = rng.random(ctx_m.shape) < self.hole_prob
            ctx_m = ctx_m & ~hole

        levels = self._build_levels(ctx_v, ctx_m)
        step_s = self._sec_per_step(freq)

        # Flatten levels from coarsest to finest, matching evaluation.
        rows_v, rows_m, rows_c, rows_t = [], [], [], []
        for j, li in enumerate(reversed(range(len(levels)))):
            wj = self.level_widths[j]
            lv_v, lv_m, lv_c = levels[li]
            rows_v.append(self._take_tail(lv_v, wj))
            rows_m.append(self._take_tail(lv_m, wj))
            rows_c.append(self._take_tail(lv_c, wj))
            cum = 1
            for r in self.ratios[:li]:
                cum *= r
            # Build timestamps for the retained tail and pad the left edge.
            n_l = lv_v.shape[0]
            off = np.arange(wj, dtype=np.float64) - float(wj - n_l)
            np.maximum(off, 0.0, out=off)
            ts_row = np.empty(self.W_out, dtype=np.float64)
            ts_row[:self.W_out - wj] = float(ts0)
            ts_row[self.W_out - wj:] = ts0 + off * (step_s * cum)
            rows_t.append(ts_row.astype(np.float32))

        x = np.stack(rows_v)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        xm = np.stack(rows_m).astype(bool)
        xc = np.stack(rows_c)
        if xc.dtype != np.float32:
            xc = xc.astype(np.float32)
        xt = self._xt_full
        xta = np.stack(rows_t).astype(np.float32)

        loc, scale = robust_stats(x.reshape(-1), xm.reshape(-1),
                                  self.norm_mode)
        if self.level_norm == "window_minmax":
            # Match the upstream window min/max normalization exactly.
            hf = np.nan_to_num(np.asarray(ctx_v, dtype=np.float32), nan=0.0,
                               posinf=0.0, neginf=0.0)
            loc = float(hf.min()) if hf.size else 0.0
            scale = max(float(hf.max()) - loc, 1e-5) if hf.size else 1.0
            xn = x
            xn -= loc
            xn /= scale
            np.clip(xn, -20.0, 20.0, out=xn)
        elif self.level_norm == "revin":
            stats = revin_level_stats(x, xm)
            # Targets and anchors use the finest-level statistics for denormalization.
            loc, scale = stats[-1]
            xn = np.stack([np.clip((x[li] - mu) / sd, -20.0, 20.0)
                           for li, (mu, sd) in enumerate(stats)]).astype(np.float32)
        else:
            xn = x
            xn -= loc
            xn /= scale
            np.clip(xn, -20.0, 20.0, out=xn)
        xn[~xm] = 0.0

        # Targets are the next H native-resolution steps after the context.
        tgt_raw = np.asarray(values[t_end:t_end + self.H], dtype=np.float32)
        tgt_m = np.asarray(valid[t_end:t_end + self.H], dtype=bool)
        if tgt_raw.shape[0] < self.H:
            pad = self.H - tgt_raw.shape[0]
            tgt_raw = np.concatenate([tgt_raw, np.zeros(pad, np.float32)])
            tgt_m = np.concatenate([tgt_m, np.zeros(pad, bool)])
        # Copy before mutating because the source may be a read-only memmap view.
        tgt = np.array(tgt_raw, dtype=np.float32, copy=True)
        tgt -= loc
        tgt /= scale
        np.clip(tgt, -20.0, 20.0, out=tgt)
        tgt[~tgt_m] = 0.0

        anchor = self._snaive_anchor(ctx_v, self._seasonality(freq))
        anchor -= loc
        anchor /= scale
        np.clip(anchor, -20.0, 20.0, out=anchor)

        # The committing reference is skipped when its weight is zero.
        copy_h = np.zeros(int(self.H), dtype=np.float32)
        if self.commit_w > 0:
            _c = self._seasonal_copy(ctx_v, ctx_m, loc, scale, freq)
            if _c is not None:
                copy_h = _c

        return (torch.from_numpy(xn),
                torch.from_numpy(xm),
                torch.from_numpy(xc),
                torch.tensor(0),
                torch.from_numpy(xt),
                torch.from_numpy(xta),
                torch.from_numpy(tgt),
                torch.tensor(float(loc), dtype=torch.float32),
                torch.tensor(float(scale), dtype=torch.float32),
                torch.from_numpy(anchor),
                torch.from_numpy(tgt_m),
                torch.from_numpy(copy_h))


def collate_shard(batch):
    """Stack batch fields along the leading dimension."""
    n = len(batch[0])
    out = []
    for i in range(n):
        xs = [b[i] for b in batch]
        out.append(torch.stack(xs) if xs[0].dim() >= 1 else torch.tensor(xs))
    return tuple(out)


class ShardBatchSampler:
    """Group rows by shard to keep batch reads contiguous within each file."""

    def __init__(self, dataset: ShardDataset, batch_size: int, seed: int,
                 skip_batches: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        n_shards = len(dataset._shard_files)
        shard_ids = np.asarray(dataset._shard_ids, dtype=np.int64)
        # Stable sorting keeps rows from the same shard contiguous.
        self.order = np.argsort(shard_ids, kind="stable")
        counts = (np.bincount(shard_ids, minlength=n_shards) if shard_ids.size
                  else np.zeros(n_shards, dtype=np.int64))
        self.bounds = np.concatenate([[0], np.cumsum(counts)])
        self.n_shards = n_shards
        self.total = len(shard_ids)
        self.epoch = 0
        self.dataset = dataset
        # Resume skips completed batches while preserving RNG progression.
        self.skip_batches = max(0, int(skip_batches))
        # Start optional page-cache prefetching.
        self.prefetcher = ShardPrefetcher()

    def __len__(self) -> int:
        return math.ceil(self.total / self.batch_size)

    def _warm_next(self, order: np.ndarray, start: int) -> None:
        """Submit the next shard files to the prefetcher without blocking."""
        pf = self.prefetcher
        names = getattr(self.dataset, "_shard_files", None)
        # Prefetch only the contiguous mmap layout.
        if pf is None or not pf.enabled or names is None \
                or not getattr(self.dataset, "fast_layout", False):
            return
        out = []
        for j in range(start, min(start + pf.lookahead, len(order))):
            sid = int(order[j])
            if 0 <= sid < len(names):
                out.append(f"{names[sid]}.values.f32.npy")
        if out:
            pf.warm(out)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = np.arange(self.n_shards)
        rng.shuffle(order)
        self._warm_next(order, 0)
        buf: list[int] = []
        n_out = 0
        for si, shard_id in enumerate(order):
            self._warm_next(order, si + 1)
            a, b = int(self.bounds[shard_id]), int(self.bounds[int(shard_id) + 1])
            if b <= a:
                continue
            buf.extend(self.order[a:b].tolist())
            while len(buf) >= self.batch_size:
                batch = buf[:self.batch_size]
                del buf[:self.batch_size]
                rng.shuffle(batch)
                n_out += 1
                if n_out <= self.skip_batches:
                    continue
                yield batch
        if buf:
            rng.shuffle(buf)
            n_out += 1
            if n_out > self.skip_batches:
                yield buf
