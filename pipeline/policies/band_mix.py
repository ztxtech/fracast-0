"""按**频段份额**重加权采 batch（论文 A.2 口径 ✓）。

官方 TinyCast 的训练语料按基准频段份额重加权：
**31 小时 / 30 次小时 / 15 日 / 8 周 / 7 月 / 6 秒**（论文 A.2）。
官方只给口径、没给代码，这里的实现是本仓库自己的落地。

为什么这么采（2026-09-15 冒烟实测）：
1. 我们的语料里 pret 有 2,842 万条序列、合成分片只有 25 万条，
   **按行均匀采样 = 被最大语料淹没** ✗ → 先按份额抽频段，再在该频段内均匀抽行 ✓。
2. 语料是 813 个分片（每片约 1 GB）的连续 mmap。**全局随机抽行**会让 512 行摊在几百个
   分片上，每个样本都要重新 `np.load` 开文件：实测 6.5 s/step、**89% 的时间在等数据** ✗
   → 每批**每个频段只进 1 个分片**，并在该分片内取**连续一段行**（mmap 顺序读 ✓）。
   一个 batch 最多碰 `len(频段)` 个文件（当前 7 个），实测 0.79 s/step（8.2 倍）✓。

抽样分布：分片按「该频段在本分片里的行数」加权，片内起点均匀、行段**环形取**
（越过片尾绕回片首），所以每条行的**边际概率完全相等** ✓ —— 与逐行均匀采样同分布，
差别只在「同批内相邻行相关」（训练不在乎，且正是它换来了顺序读 ✓）。

⚠️ **撑不住的份额要封顶**（2026-09-15 定的口径）：份额是「官方语料里的比例」，我们的语料
未必填得满。本语料 `second` 段只有 **1 行**（`pret/solar_power` 的 4S 序列，739 万点），
按 6% 份额 = 每批 2048 里有 123 个窗口出自这**一条**序列 ✗ —— 那不是官方配方，是数据缺口
的伪影。所以：某频段「单分片最大行数 < 份额 × batch_size」时，把它的份额压到
`最大行数 / batch_size`，多出来的份额**按现有比例再分配给其他频段**（迭代到不再越界 ✓）。
分配动过的频段会在构造时打印，属于明确的实现偏离。

配置（`data.band_mix`）：
    shares: {second: 0.06, subhour: 0.30, hour: 0.31, day: 0.15, week: 0.08, month: 0.07}
    其余频段（季/年/other 等）份额 = 1 - Σshares（自动平分 ✓）
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np


def band_of(freq: str) -> str:
    """freq 字符串 → 频段名（未识别的进 "other" ✓）。

    口径按语料里实测出现的 freq 清单写死顺序（`data/corpus_fast/*/p*/freqs.txt`：
    `? 10T 15T 30T 4S 5T 6H A-DEC D H M MS Q-DEC T W W-SUN` ✓）——
    **"MS"（月初）必须先于 `endswith('S')` 判**，否则会被当成秒级 ✗。
    """
    f = str(freq).strip()
    if not f:
        return "other"
    if f in ("M", "MS", "ME", "BM", "BMS") or f.startswith("M-"):
        return "month"
    # 年/季别名也以 S 结尾（AS/YS/QS），必须先于秒级判断 ✓
    if f in ("A", "AS", "YS") or f.startswith("A-") or f.startswith("Y"):
        return "other"
    if f.startswith("Q"):
        return "other"
    if f.endswith("S"):                                # 4S / 10S / S —— 秒级
        return "second"
    if f.endswith("T") or f.lower().endswith("min"):   # T / 5T / 15T / 30T / 10min
        n = int(f[:-1]) if f[:-1].isdigit() else 1
        return "subhour" if n < 60 else "hour"
    if f.endswith("H"):                                # H / 6H
        return "hour"
    if f.endswith("D"):                                # D / A-DEC 之外
        return "day"
    if "W" in f:                                       # W / W-SUN
        return "week"
    return "other"                                     # ? / A-DEC / Q-DEC …


def _band_shares(sizes: "OrderedDict[str, int]", want: dict, batch_size: int
                 ) -> "OrderedDict[str, float]":
    """频段 → 归一化份额（点名用配置值 / 其余平分剩余 / 撑不住的封顶再分 ✓）。

    `sizes` = 各频段**单个分片里的最大行数**（一批只进一个分片，所以它就是一批能取到的
    不重复行数上限 ✓）；`want` = 配置里的份额。
    """
    named = {b: float(want[b]) for b in sizes if b in want}
    unnamed = [b for b in sizes if b not in want]
    rest = max(0.0, 1.0 - float(sum(named.values())))
    if unnamed:
        named.update({b: rest / len(unnamed) for b in unnamed})
    keep = OrderedDict((b, w) for b, w in named.items() if w > 0)
    if not keep:
        raise ValueError("band_mix：所有频段份额都是 0 ✗")

    capped: dict[str, float] = {}
    for _ in range(16):                     # 迭代：封顶 → 余量按比例再分 → 再看有没有越界
        over = [b for b, w in keep.items()
                if w > sizes[b] / float(batch_size) + 1e-12 and b not in capped]
        if not over:
            break
        for b in over:
            capped[b] = keep[b]
            keep[b] = sizes[b] / float(batch_size)
        free = [b for b in keep if b not in capped and keep[b] > 0]
        left = max(0.0, 1.0 - float(sum(keep.values())))
        tot = float(sum(keep[b] for b in free))
        if not free or tot <= 0:
            break
        for b in free:
            keep[b] += left * keep[b] / tot
    if capped:
        print("[band] 频段份额封顶（语料撑不满，余量按比例分给其他频段）："
              + " ".join(f"{b} {100 * w:.2f}%→{100 * keep[b]:.2f}%"
                         for b, w in capped.items()), flush=True)

    total = float(sum(keep.values()))
    return OrderedDict((b, w / total) for b, w in keep.items())


class _BandPool:
    """一个频段的行池：按分片分组，支持「挑分片 → 片内取一段」 ✓。

    内存口径：行号统一 int32（语料总行数 ≪ 2³¹），比 int64 省一半 ✓；
    `flat` 按分片连续排列，`starts[s]:starts[s+1]` 就是分片 s 在本频段的行号切片 ✓。
    """

    __slots__ = ("flat", "starts", "shards", "counts", "p", "max_rows")

    def __init__(self, rows: np.ndarray, shard_ids: np.ndarray, n_shards: int):
        sid = shard_ids[rows]
        order = np.argsort(sid, kind="stable")
        self.flat = rows[order].astype(np.int32, copy=False)
        counts = np.bincount(sid, minlength=n_shards)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.shards = np.flatnonzero(counts).astype(np.int64)
        self.counts = counts[self.shards]
        self.p = self.counts / float(self.counts.sum())
        self.max_rows = int(self.counts.max())      # 单个分片里的最大行数（份额封顶用 ✓）

    def draw(self, need: int, rng: np.random.Generator) -> np.ndarray:
        """取一段 `need` 行：先按行数加权挑分片，再在片内环形取连续行 ✓。"""
        s = int(self.shards[int(rng.choice(self.shards.size, p=self.p))])
        a, b = int(self.starts[s]), int(self.starts[s + 1])
        n = b - a
        off = int(rng.integers(0, n))
        pos = a + (off + np.arange(int(need))) % n
        return self.flat[pos]


class BandMixSampler:
    """按频段份额抽 batch 的 batch_sampler（接口与 `ShardBatchSampler` 一致 ✓）。"""

    def __init__(self, dataset, batch_size: int, seed: int,
                 grad_accum_steps: int, cfg: dict, skip_micro: int = 0):
        data = cfg.get("data") or {}
        train = cfg.get("train") or {}
        mix = data.get("band_mix") or {}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.micro_steps = max(1, int(train.get("total_steps") or 1)) * max(
            1, int(grad_accum_steps))
        # 断点续跑（用户 2026-09-17）：开头的 skip_micro 个 micro batch 照常抽（RNG
        # 逐位对齐 ✓）但不产出 —— 数据流精确接在断点上，既不重读旧数据、也不改后续顺序 ✓。
        self.skip_micro = max(0, min(int(skip_micro), self.micro_steps))

        # 频率编号 → 频段编号（int8），再散射到行：不做 3,570 万个 Python 字符串 ✗
        codes = np.empty(max(len(dataset._freqs), 1), dtype=np.int8)
        code_of: "OrderedDict[str, int]" = OrderedDict()
        for i, f in enumerate(dataset._freqs):
            b = band_of(f)
            if b not in code_of:
                if len(code_of) > 100:
                    raise ValueError("band_mix：频段种类异常多（>100）✗")
                code_of[b] = len(code_of)
            codes[i] = code_of[b]
        band_codes = codes[np.asarray(dataset._freq_ids, dtype=np.int64)]

        shard_ids = np.asarray(dataset._shard_ids)
        n_shards = len(dataset._shard_files)
        pools: "OrderedDict[str, _BandPool]" = OrderedDict()
        for b, code in code_of.items():
            rows = np.flatnonzero(band_codes == code)
            if rows.size:
                pools[b] = _BandPool(rows, shard_ids, n_shards)
        if not pools:
            raise ValueError("band_mix：语料里没有可采样的行 ✗")

        sizes = OrderedDict((b, pool.max_rows) for b, pool in pools.items())
        shares = _band_shares(sizes, {str(k): float(v)
                                      for k, v in (mix.get("shares") or {}).items()},
                              self.batch_size)
        self.bands = list(shares.keys())
        self.p = np.asarray([shares[b] for b in self.bands], dtype=np.float64)
        self.pools = [pools[b] for b in self.bands]

        n_rows = int(sum(int(pool.flat.size) for pool in self.pools))
        print(f"[band] {len(self.bands)} 段 / {n_rows:,} 行 → "
              + " ".join(f"{b}({pool.flat.size:,}行,{100 * p:.2f}%,{pool.shards.size}片)"
                         for b, pool, p in zip(self.bands, self.pools, self.p)),
              flush=True)

    def __len__(self) -> int:
        return self.micro_steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        # 每批各频段行数固定（分层抽样：批间组成不抖、份额精确 ✓），零头给份额最大的频段
        need = np.floor(self.p * self.batch_size).astype(np.int64)
        need[int(np.argmax(self.p))] += self.batch_size - int(need.sum())
        for i in range(self.micro_steps):
            out = np.empty(self.batch_size, dtype=np.int32)
            at = 0
            # 按频段成组填充且**不洗牌**：批内顺序 = 文件里的顺序 → mmap 顺序读 ✓
            # （损失的样本维顺序不影响结果，只影响 fp 归约的末位 ✓）
            for bi in range(len(self.bands)):
                k = int(need[bi])
                if k > 0:
                    out[at:at + k] = self.pools[bi].draw(k, rng)
                    at += k
            if i < self.skip_micro:
                continue          # 只推进 RNG，不产出 batch（续跑对齐用 ✓）
            yield out.tolist()


def make_band_mix_sampler(dataset, batch_size: int, seed: int,
                          grad_accum_steps: int, cfg: dict,
                          skip_micro: int = 0):
    """工厂口（`build_train_loaders(sampler_factory=...)` 用 ✓）。"""
    return BandMixSampler(dataset, batch_size, seed, grad_accum_steps, cfg,
                          skip_micro=skip_micro)
