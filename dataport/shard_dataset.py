"""ShardDataset：分片训练端（memmap 优先，npz 兼容），语义与评测端严格对齐。

对齐基准：`fractal_tsfm/gifteval_predictor.py::_forecast_quantiles`。
训练样本的构造必须和评测时喂给模型的东西一致，否则训练目标与评测指标脱节。

对齐点（逐条对照评测端实现）：
1. 上下文 = 目标窗口之前的 ctx_len = W x prod(ratios) 步原始历史（最多 4096）。
2. 金字塔自底向上构建（级 0 = 原始分辨率），展平顺序 **粗级在前、最细级在后**
   （评测端 `for lv in reversed(levels)`）。
3. 每级窗口取末 W 步，不足左侧补 0 + mask=False；时间戳补 edge。
4. 归一化 = `RobustNorm(mode)` 在展平上下文上 fit（只统计有效点），
   并做 ±max_abs 截断；`loc/scale` 与评测端一致，用于还原预测。
5. `ts_norm` = 每级 `arange(W)/W`；`t_abs` = 真实秒级时间戳（日历相位用）。
6. `coverage`：级 0 = 有效掩码本身，聚合级 = 箱内有效观测比例（与评测端一致）。
7. 目标 = **原始分辨率**未来 H 步（不与上下文重叠），缺失处 mask=False。

旧版本的问题：
- 目标取了 `levels[-1]`（最粗级，池化 16 倍）而非原始分辨率；
- 缺失位置填 0 且 `tgt_mask` 未参与 loss，等于让模型去拟合假 0；
- 级别展平顺序、ts_norm/t_abs、归一化统计量都与评测端不一致。
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
    """RevIN 逐级统计量（ts-kim/RevIN RevIN.py L32-35：mean/std，有偏 var+eps，detach）。

    x [L, W]，返回 [(mu, sd), ...]（顺序与级一致，粗级在前）。
    """
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
    """复刻 model.RobustNorm.fit 的统计量（逐样本 numpy 版）。

    `_masked_quantile` 的实现是「排序后取 floor(cnt*q) 位」，不是线性插值，
    这里必须一致，否则归一化口径会有系统性偏差。
    """
    m = flat_m.astype(bool)
    if not m.any():
        return 0.0, 1.0
    # 无效位置可能是 inf（不只是 NaN）：inf²×0 仍是 inf，会把 scale 污染成 inf ✗
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
        # 官方版为窗口因果统计的末值 = 全窗有效点 mean/std
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
    """分片数据集：memmap 随机访问 + 评测对齐的窗口构造。

    train=True 时每个 epoch 为每条序列随机抽一个切点（窗口多样）；
    train=False 时用固定切点（可复现的验证损失）。
    """

    def __init__(self, shard_dir, cfg, train=True, cache_shards: int = 64):
        self.dir = Path(shard_dir)
        compact_path = self.dir / "index_compact.npz"
        if not compact_path.exists():
            raise FileNotFoundError(
                f"缺少紧凑索引 {compact_path}，先跑 script/tools/make_compact_index.py")
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
        # 布局探测：memmap 版（<stem>.values.npy）优先，否则退回压缩 npz
        probe = self.dir / (self._stem(shard_files[0]) + ".values.npy")
        self.fast_layout = probe.exists()
        self._ts_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _split_mask(n: int, cfg, train: bool) -> np.ndarray:
        """训练/验证切分：按位置稳定切（跨进程一致）。"""
        frac = cfg["data"]["val_fraction"]
        pos = np.arange(n) % 10
        cut = round((1 - frac) * 10)
        return pos < cut if train else pos >= cut

    def _init_windows(self, cfg, train: bool, cache_shards: int) -> None:
        """窗口/归一化/增强准备（与语料布局无关，子类共用 ✓）。

        调用方必须先设好 `_shard_files`/`_freqs`/`_shard_ids`/`_row_ids`/`_freq_ids` ✓。
        """
        self.cfg = cfg
        self.train = train
        self.H = int(cfg["model"]["horizon"])
        self.ratios = list(cfg["pyramid"]["ratios"])
        self.L = len(self.ratios) + 1
        # 逐级 token 宽度（粗级在前；`model.level_widths` 缺省 = 老行为：全级同宽 ✓）。
        # `self.W` 保持历史语义 = **最细级（近端）**宽度；网格宽 `W_out = max(widths)`。
        self.level_widths = resolve_level_widths(
            cfg["model"].get("level_widths", cfg["model"]["W"]), self.L)
        self.W = self.level_widths[-1]
        self.W_out = max(self.level_widths)
        # 上下文上限：默认 4096 = 历史行为逐位一致（历史 config_used.yaml 无此键）。
        # `data.ctx_len_cap: 0`（或负数）= 不设上限，用满金字塔覆盖到的全部历史。
        # 背景（2026-09-11）：语料中位序列 8,761 步、旧的 4096 截断丢掉了 58.1% 的数据点；
        # 而 v27 的 W×prod(ratios) = 2048 < 4096，说明它连上限都没碰到、只用了 2048 步。
        _cap = int((cfg.get("data") or {}).get("ctx_len_cap", 4096) or 0)
        _span = level_span(self.level_widths, self.ratios)
        self.ctx_len = _span if _cap <= 0 else min(_span, _cap)
        self.norm_mode = cfg["model"].get("context_norm", "robust")
        # revin = 逐金字塔层级做 RevIN（mean/std）；robust_global = 原实现（整窗一个统计量）
        self.level_norm = cfg["data"].get("level_norm", "robust_global")
        # ── 季节性复制参照（committing loss 用；官方 TinyCast losses.py L130-179）──
        # 默认 0 = 不计算 → 行为与历史逐位一致。
        self.commit_w = float((cfg.get("train") or {}).get("commit_w", 0.0) or 0.0)
        self._sf_cache: dict[str, float] = {}
        self._sf_missing: set[str] = set()
        # 加速缓存（2026-09-14 用户「按最快的来」）：
        #   gluonts.get_seasonality 每样本都走一次 pandas offset（实测 ~0.05 ms/样本 ✗）
        #   → 按 freq 缓存；freq_to_seconds 同理（字符串解析）。
        #   xt 是常量 [L, W] → 预先算好一次（collate 会 stack 成新张量，共享只读常量安全 ✓）。
        self._s_cache: dict[str, int] = {}
        self._sec_cache: dict[str, float] = {}
        # 逐级时间轴：每级自己的 arange(w)/w，左侧补零到网格宽（均匀宽度下 = 旧值 ✓）
        self._xt_full = np.stack([
            np.pad(np.arange(w, dtype=np.float32) / np.float32(w),
                   (self.W_out - w, 0)) for w in self.level_widths])
        self.hole_prob = float(cfg["data"].get("hole_prob", 0.0)) if train else 0.0
        # ── 原始窗口模式（官方 TinyCast rollout 对齐，2026-09-15 用户定）──
        # 训练侧只给「原始分辨率窗口 + 掩码 + 季节 scale」，金字塔与归一化挪到 GPU 上
        # 按块重建（每块用自己那段上下文 → 官方 _rollout_loss 口径 ✓）。
        # 验证集仍走旧的金字塔样本（val_pinball 只作廉价曲线、不参与选型 ✓）。
        self.window_mode = str((cfg.get("data") or {}).get("window_mode", "levels"))
        self.ar_chunks = int((cfg.get("train") or {}).get("ar_chunks", 1) or 1)
        self.emit_window = bool(train and self.window_mode == "raw")
        self.tgt_span = self.H * (self.ar_chunks if self.emit_window else 1)
        self.win_len = self.ctx_len + self.tgt_span
        # 逐行季节 scale 覆盖（合成分片带官方 scale_factors.f32 侧车；没有则为 None）
        self._row_scales = None
        self.seed = int(cfg["data"].get("seed", 42)) + (0 if train else 1)
        self._cache_shards = cache_shards
        self._cache: OrderedDict[str, dict] = OrderedDict()

    # ---- 布局无关的读取 ----
    def _sample_rng(self, i: int) -> np.random.Generator:
        """逐样本 RNG：种子由 `(seed, 行下标)` 决定 ✓。

        为什么不用一个实例级 `self.rng`（2026-09-17 改）：那是**调用序**驱动的 ——
        多 worker 下「谁先取到哪一行」由调度决定，断点续跑又**不会重放被跳过的样本**
        ⇒ 同一行拿到的随机切点/掩码与中断前不同，续跑立刻偏离原轨迹 ✗
        （实测：同一条流，step4 loss 0.2143 → 0.2141，到 step6 放成 2.7%）。
        改成下标派生后，样本内容是 `(seed, 下标)` 的纯函数 —— 与 worker 数、调度顺序、
        续跑位置全部无关 ✓（抽样分布不变：仍然是各自独立的均匀抽样 ✓）。
        """
        return np.random.default_rng((int(self.seed), int(i)))

    def _stem(self, fname: str) -> str:
        return fname[:-4] if fname.endswith(".npz") else fname

    def _load_row(self, shard_file: str, row: int
                  ) -> tuple[np.ndarray, np.ndarray, int]:
        """返回 (values[T], valid[T], ts0)。只读通道 0。"""
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
        """自底向上建金字塔。返回 [(v[T_l], m[T_l], cov[T_l])]，级 0 = 原始分辨率。"""
        levels = [(values, valid, valid.astype(np.float32))]
        for r in self.ratios:
            pv, pm, _ = levels[-1]
            T = (pv.shape[0] // r) * r
            if T < r:
                break
            v2 = pv[:T].reshape(T // r, r)
            m2 = pm[:T].reshape(T // r, r)
            cnt = m2.sum(axis=1)
            # 先把无效位清零再求和：NaN × 0 仍是 NaN，会把只含少量缺失的块整块污染成 NaN ✗
            # （2026-09-13 实测：chronos 语料有 NaN，池化层的 m=True 位置于是携带 NaN 进训练）
            pv0 = np.where(m2, v2, 0.0)
            agg = np.where(cnt > 0, pv0.sum(axis=1) / np.maximum(cnt, 1), 0.0)
            levels.append((agg.astype(np.float32), cnt > 0, cnt / float(r)))
        # 级数不足时在最前面复制级 0（与评测端 while len(levels) < L 同口径）
        while len(levels) < self.L:
            levels.insert(0, levels[0])
        return levels

    def _seasonal_scale(self, freq: str):
        """官方 scale.seasonal_scale_factor(freq, domain=None) 的缓存包装。

        偏离 #1：domain=None（训练期无域信息，仅影响 D/W 频率的 lag 选择）。
        未实现的频率记入 _sf_missing 并返回 None（该样本不加 committing 项）。
        """
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
        """季节复制参照 [H]，**归一化空间**（与 tgt 同尺度）。失败返回 None。

        官方 seasonal_copy_baseline 原样调用；偏离见模块顶部 #1/#2/#3。
        """
        import torch as _t
        from module.losses.tinycast import seasonal_copy_baseline
        sf = self._seasonal_scale(freq)
        if sf is None:
            return None
        v = np.asarray(ctx_v, dtype=np.float32)
        m = np.asarray(ctx_m, dtype=bool)
        if v.size < 4 or not m.any():
            return None
        if not m.all():                     # 偏离 #3：前向填充
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
        """取本级末 `width` 个 token，再左侧补零到网格宽 `self.W_out` ✓。

        最细级宽度 = W_out 时不补零 → 与历史实现逐位一致（均匀宽度下完全等价 ✓）。
        """
        keep = min(int(width), arr.shape[0])
        tail = arr[arr.shape[0] - keep:]
        lead = self.W_out - keep
        if lead <= 0:
            return np.ascontiguousarray(tail)
        pad = np.zeros((lead,) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([pad, tail], axis=0)

    def _pick_t_end(self, values: np.ndarray, valid: np.ndarray,
                    rng: np.random.Generator) -> int:
        """选上下文结束位置（也是目标起点）。"""
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return max(1, values.shape[0] // 2)
        last = int(idx[-1]) + 1
        if not self.train:
            # 验证：固定用「末尾 H 步作为目标」的切点，保证可复现
            return max(1, last - self.H) if last > self.H else max(1, last // 2)
        lo = 1
        hi = max(lo + 1, last)
        for _ in range(8):
            t = int(rng.integers(lo, hi))
            cm = valid[max(0, t - self.ctx_len):t]
            # rollout 模式要整段 K×p 目标都有观测；单块模式 = p 步 ✓
            tgt = valid[t:t + self.tgt_span]
            if cm.sum() >= 8 and tgt.sum() >= 8:
                return t
        return max(1, last - self.H) if last > self.H else max(1, last // 2)

    def _row_scale(self, row: int, freq: str) -> float:
        """逐行季节 scale：优先侧车（合成分片的官方 scale_factors.f32），否则由 freq 推 ✓。"""
        rs = self._row_scales
        if rs is not None and 0 <= row < rs.shape[0]:
            s = float(rs[row])
            if np.isfinite(s) and s > 0:
                return s
        s = self._seasonal_scale(freq)
        return float(s) if s else 1.0

    def _window_sample(self, values, valid, t_end: int, row: int, freq: str):
        """原始窗口样本：值 + 掩码 + 季节 scale（官方 TinyCast rollout 口径 ✓）。

        形状 [ctx_len + K×p]；上下文不足时在**左侧**补 0（目标贴右端，与旧路径同口径 ✓）。
        NaN/Inf 一次性填 0（官方 `_unpack_batch` 同口径 ✓）。
        """
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
        """freq → 季节周期 S（按 freq 缓存；get_seasonality 内部走 pandas offset，很贵 ✗）。"""
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
        """freq → 每步秒数（按 freq 缓存，避开每样本重复的字符串解析 ✓）。"""
        v = self._sec_cache.get(freq)
        if v is None:
            v = float(freq_to_seconds(freq))
            self._sec_cache[freq] = v
        return v

    def _snaive_anchor(self, ctx: np.ndarray, S: int) -> np.ndarray:
        """SNaive 锚 [H]，与 gifteval_predictor._snaive_anchor_ctx 同口径。

        语义：i_k = T - S + (k mod S)；窗口有 NaN/inf 时回退到「i_k 之前最近的有效观测」；
        连前面也没有有效观测时，回退到「锚自身前面最近的有效值」，再不行给 0。

        向量化（2026-09-14 用户「按最快的来」）：H 步 Python 循环 + while 回退是单样本最大
        热点（实测 ~0.14 ms/样本 ≈ 全流程 1/5）。两条路径：整周期无 NaN → 直接 gather（常见 ✓）；
        有 NaN → 用「前向有效索引」一次向量化回退 ✓，语义与逐点循环逐位一致。
        """
        H = self.H
        T = ctx.shape[0]
        if S < 1:
            S = 1
        # ★ 必须取模：SNaive = 重复最后一个完整周期（statsforecast SeasonalNaive 语义）。
        #   原实现写 `T - 1 - S + k`（无 mod）→ k >= S 时索引越界、回退末值，
        #   于是 H=96/S=24 的锚后 72 步退化成平线（2026-09-12 定位）。
        anchor = np.empty(H, dtype=np.float32)
        if S <= T:
            pos = (T - S) + (np.arange(H, dtype=np.int64) % S)
            np.take(ctx, pos, out=anchor)
            if not np.isfinite(ctx[T - S:]).all():
                return self._fill_anchor_nan(ctx, anchor, pos)
            return anchor
        # S > T（序列比一个周期还短）：pos 可能为负 → 回退链。
        #   ★ 旧实现在这里读的是 `np.empty` 未初始化内存（k=0 可能把垃圾值当锚 ✗）；
        #     现在显式填 NaN → 回退链拿不到有效值就给 0（确定性 ✓）。
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
        """锚里的 NaN/inf → 回退到「pos 之前最近的有效观测」（向量化，替代逐点 while ✓）。"""
        T = ctx.shape[0]
        fin = np.isfinite(ctx)
        if not fin.any():
            anchor.fill(0.0)
            return anchor
        idx = np.where(fin, np.arange(T), -1)
        np.maximum.accumulate(idx, out=idx)      # idx[i] = 最近 ≤ i 的有效位置（-1 = 前面无）
        src = idx[pos]
        ok = src >= 0
        if ok.any():
            anchor[ok] = ctx[src[ok]]
        for k in np.flatnonzero(~ok):            # 极少见：入口前面全是 NaN
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

        rng = self._sample_rng(i)      # 逐样本 RNG：与 worker 调度 / 续跑位置无关 ✓

        t_end = self._pick_t_end(values, valid, rng)
        # 官方 rollout 口径：训练侧直接给原始窗口，金字塔与归一化在 GPU 上按块重建 ✓
        if self.emit_window:
            return self._window_sample(values, valid, t_end, int(i), freq)
        t0 = max(0, t_end - self.ctx_len)
        ctx_v = np.asarray(values[t0:t_end], dtype=np.float32)
        ctx_m = np.asarray(valid[t0:t_end], dtype=bool)
        if ctx_v.size == 0:                       # 极端短序列兜底
            ctx_v = np.zeros(1, dtype=np.float32)
            ctx_m = np.zeros(1, dtype=bool)

        # 打洞增强（只作用于训练，且只打上下文）
        if self.train and self.hole_prob > 0 and ctx_m.any():
            hole = rng.random(ctx_m.shape) < self.hole_prob
            ctx_m = ctx_m & ~hole

        levels = self._build_levels(ctx_v, ctx_m)
        step_s = self._sec_per_step(freq)

        # 展平顺序：粗级在前、最细级在后（与评测端 reversed(levels) 一致）
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
            # 时间戳：末 wj 步（不足左侧补 ts0）—— 下标整体下移 (wj-n) 再截到 >= 0；
            # 网格左侧 padding 位取 ts0（与「时间戳补 edge」口径一致 ✓）。
            n_l = lv_v.shape[0]
            off = np.arange(wj, dtype=np.float64) - float(wj - n_l)
            np.maximum(off, 0.0, out=off)
            ts_row = np.empty(self.W_out, dtype=np.float64)
            ts_row[:self.W_out - wj] = float(ts0)
            ts_row[self.W_out - wj:] = ts0 + off * (step_s * cum)
            rows_t.append(ts_row.astype(np.float32))

        x = np.stack(rows_v)                             # [L, W]
        if x.dtype != np.float32:                        # values 已是 f32 → 省掉整块拷贝 ✓
            x = x.astype(np.float32)
        xm = np.stack(rows_m).astype(bool)
        xc = np.stack(rows_c)
        if xc.dtype != np.float32:
            xc = xc.astype(np.float32)
        xt = self._xt_full                               # 常量 [L, W]（collate 会 stack 拷走 ✓）
        xta = np.stack(rows_t).astype(np.float32)

        loc, scale = robust_stats(x.reshape(-1), xm.reshape(-1),
                                  self.norm_mode)
        if self.level_norm == "window_minmax":
            # 官方 WindowMinMax（`tinycast/normalization.py`）：整条原生上下文 min/max，
            # nan_to_num 先做、range 下限 1e-5 —— 与训练侧 `_rollout_loss` 同口径 ✓
            hf = np.nan_to_num(np.asarray(ctx_v, dtype=np.float32), nan=0.0,
                               posinf=0.0, neginf=0.0)
            loc = float(hf.min()) if hf.size else 0.0
            scale = max(float(hf.max()) - loc, 1e-5) if hf.size else 1.0
            xn = x                                       # x 之后不再用 → 原地归一化省分配 ✓
            xn -= loc
            xn /= scale
            np.clip(xn, -20.0, 20.0, out=xn)
        elif self.level_norm == "revin":
            stats = revin_level_stats(x, xm)
            # 目标/锚在原始分辨率上，用最细级（rows 最后一级）统计量，对应 RevIN denorm
            loc, scale = stats[-1]
            xn = np.stack([np.clip((x[li] - mu) / sd, -20.0, 20.0)
                           for li, (mu, sd) in enumerate(stats)]).astype(np.float32)
        else:
            xn = x                                       # x 之后不再用 → 原地归一化省分配 ✓
            xn -= loc
            xn /= scale
            np.clip(xn, -20.0, 20.0, out=xn)
        xn[~xm] = 0.0

        # 目标：原始分辨率的未来 H 步（严格不与上下文重叠）
        tgt_raw = np.asarray(values[t_end:t_end + self.H], dtype=np.float32)
        tgt_m = np.asarray(valid[t_end:t_end + self.H], dtype=bool)
        if tgt_raw.shape[0] < self.H:
            pad = self.H - tgt_raw.shape[0]
            tgt_raw = np.concatenate([tgt_raw, np.zeros(pad, np.float32)])
            tgt_m = np.concatenate([tgt_m, np.zeros(pad, bool)])
        # 原地版：tgt_raw 可能是只读 memmap 视图 → 先拷一份再算 ✓（数值与旧实现一致）
        tgt = np.array(tgt_raw, dtype=np.float32, copy=True)
        tgt -= loc
        tgt /= scale
        np.clip(tgt, -20.0, 20.0, out=tgt)
        tgt[~tgt_m] = 0.0

        anchor = self._snaive_anchor(ctx_v, self._seasonality(freq))   # 新数组 ✓
        anchor -= loc
        anchor /= scale
        np.clip(anchor, -20.0, 20.0, out=anchor)

        # committing 参照（默认 commit_w=0 → 只返回零张量，不计算、不影响数值）
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
                torch.from_numpy(copy_h))          # [H] 季节复制参照（归一化空间）


def collate_shard(batch):
    """默认按特征轴堆叠；tgt_mask [B,H] 一并返回。"""
    n = len(batch[0])
    out = []
    for i in range(n):
        xs = [b[i] for b in batch]
        out.append(torch.stack(xs) if xs[0].dim() >= 1 else torch.tensor(xs))
    return tuple(out)


class ShardBatchSampler:
    """按分片打包 batch（同一分片一次读入，避免跨文件随机 seek 抖动 ✓）。

    numpy 分组（2026-09-13 改）：旧实现是 `list[list[int]]` —— 新语料 53.8M 行时
    光分组索引就要 2+ GB 常驻 ✗。改成「稳定 argsort + 边界数组」后内存降到约 1/8 ✓，
    且组内顺序、片间随机顺序、batch 组成与旧实现**逐位一致** ✓。
    """

    def __init__(self, dataset: ShardDataset, batch_size: int, seed: int,
                 skip_batches: int = 0):
        self.batch_size = batch_size
        self.seed = seed
        n_shards = len(dataset._shard_files)
        shard_ids = np.asarray(dataset._shard_ids, dtype=np.int64)
        # 稳定排序：同分片的行下标连续，组内顺序 = 数据集顺序（与旧实现一致 ✓）
        self.order = np.argsort(shard_ids, kind="stable")
        counts = (np.bincount(shard_ids, minlength=n_shards) if shard_ids.size
                  else np.zeros(n_shards, dtype=np.int64))
        self.bounds = np.concatenate([[0], np.cumsum(counts)])
        self.n_shards = n_shards
        self.total = len(shard_ids)
        self.epoch = 0
        self.dataset = dataset
        # 断点续跑（用户 2026-09-17）：跳过开头 skip_batches 个 batch；跳过的那些
        # 仍照常走 `rng.shuffle(batch)`，保证 RNG 逐位对齐 ✓。
        self.skip_batches = max(0, int(skip_batches))
        # 分片级预读（见 dataport/prefetch.py）：入口开线程，不改任何样本 ✓
        self.prefetcher = ShardPrefetcher()

    def __len__(self) -> int:
        return math.ceil(self.total / self.batch_size)

    def _warm_next(self, order: np.ndarray, start: int) -> None:
        """把 order[start:start+lookahead] 的分片文件交给预读线程 ✓（不阻塞、重复自动去重）。

        为什么在这里：sampler 的分片顺序就是 worker 接下来要读的顺序 ✓；进入某分片时
        它后面 lookahead 个分片正好还有几秒~十几秒才被读，够顺序灌完页缓存 ✓。
        """
        pf = self.prefetcher
        names = getattr(self.dataset, "_shard_files", None)
        # 只对连续 mmap 布局预读（旧 npz 布局的文件名不是 `<stem>.values.f32.npy` ✗）
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
        self._warm_next(order, 0)          # 进场先预读头两个分片 ✓
        buf: list[int] = []
        n_out = 0                         # 已产出的 batch 数（含跳过的 ✓）
        for si, shard_id in enumerate(order):
            self._warm_next(order, si + 1)  # 每进一个分片，把后面 lookahead 个排上 ✓
            a, b = int(self.bounds[shard_id]), int(self.bounds[int(shard_id) + 1])
            if b <= a:      # 该分片在本 split（train/val）里没有行 → 跳过（与旧实现一致 ✓）
                continue
            buf.extend(self.order[a:b].tolist())
            while len(buf) >= self.batch_size:
                batch = buf[:self.batch_size]
                del buf[:self.batch_size]
                rng.shuffle(batch)
                n_out += 1
                if n_out <= self.skip_batches:
                    continue   # 只推进 RNG，不产出 batch（续跑对齐用 ✓）
                yield batch
        if buf:
            rng.shuffle(buf)
            n_out += 1
            if n_out > self.skip_batches:
                yield buf
