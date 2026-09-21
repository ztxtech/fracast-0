"""把原始 HF arrow/parquet 语料转成连续 mmap 格式。

## 设计决策（都有依据）
1. **连续 mmap + 全局 offsets**：所有序列首尾相接写进 `values.f32`，
   `offsets.npy`(int64, N+1) 给出每条的 [start,end)。
   取任意一条 = `values[off[i]:off[i+1]]` —— **O(1)、无 open/seek**。
   （我们踩过的坑：逐条 open/read 让 12 个 worker 只跑到 1 核；这是本节要根除的。）
2. **不设任何上下文/长度限制**：
   - **不truncate**（保留最长 7,397,222 步的原序列）；
   - 不写 `ctx_len_cap`；`max_seq_len` 由**训练配置**决定，与数据格式无关。
3. **按数据集连续存放**（不再按长度分箱切碎）：同一数据集的序列落在同一分片内，
   顺序读时局部性好；采样要的长度分桶另行用**索引**表达（`buckets.npz`），
   **不改动数据排布** —— 这样"分桶采样"与"连续 IO"二者不再互相牺牲。
4. **float32 无损**：f16 会把 1.976e20 这类极值截断（我们踩过），
   而读取速度瓶颈是"随机 vs 连续"而非"4 字节 vs 2 字节"；先用无损格式，
   若后续实测 IO 受限再评估 f16。
5. **分片大小** ~8 GB：多 worker 并行读不同分片，且单个文件仍在页缓存友好范围内。

## 边界
本文件 = **格式实现**（读取器 + 分片写出）✓；**流程**在 `pipeline/build_corpus.py` ✓
（那边只管 config → 参数，不含格式细节）。包内唯一调用方是 `pipeline/build_corpus.py` ✓；
**本文件没有 CLI** ✗（主入口只有根目录 main.py ✓）。

## 用法（全部走配置 ✓）
    # 一份配置：src / out / shard_gb / [dry] / [max_files]
    env -u PYTHONPATH .venv/bin/python main.py config/corpus/<名>.yaml
    # 并行 = 网格：_run.grid: {part: [0..31]} + balance=bytes → --workers 32
    env -u PYTHONPATH .venv/bin/python main.py config/corpus/<名>.yaml --workers 32
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np


def _ts_epoch(t) -> int:
    """时间 → epoch 秒（datetime / pandas.Timestamp / 数字 / 字符串 / 列表；失败返回 0）。"""
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
    """从 HF arrow 文件逐条产出 (item_id, freq, values)。
    契约（与我们既有的 convert_synth_to_arrow.py 一致）：
      target: fixed_size_list<list<float>>[2]（通道维在前）；
      start: timestamp[s]；freq / item_id 为字符串列。
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
    # ★ start 列（起始时间戳）：日历特征需要；缺失时用 0（不静默丢，只在文件级打印一次）
    if "start" in cols:
        st_col = table.column("start").to_pylist()
    elif "timestamp" in cols:
        st_col = table.column("timestamp").to_pylist()
    else:
        st_col = [0] * table.num_rows
        print(f"    [warn] {path.split('/')[-1]}: 无 start/timestamp 列 → ts 记 0", flush=True)
    if "target" not in cols:
        return
    col = table.column("target")
    # ★ 零拷贝快速路径（2026-09-13 二次复检加）：target 是「每行一条序列」的 list 列
    #   （fixed_size_list<list<float>>[C] 或 list<float>）时，旧路径 to_pylist() 会把每条
    #   序列变成 Python list —— cmip6_1850 每行 53 通道、weatherbench 每条 35 万点，
    #   实测慢到不可用。这里直接切 arrow 的 offsets/values 缓冲（O(1) 取行、零拷贝）。
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
                # 保留 NaN 原值（掩码由适配器派生）
                if a.size >= 16:
                    yield str(iid[i]), f, a, ts0
            else:
                # ★ 多变量必须**逐通道展开成独立序列**：抽查发现 PEMS/建筑/气候等数据集是
                #   [C, T]（C 可达数百），只取第 0 通道会丢掉绝大部分数据（我们之前就吃过这个亏）。
                #   每个通道作为一条独立序列产出（item_id 加后缀以保持可追溯）。
                for c in range(n_chan):
                    j = i * n_chan + c
                    row = _as_f32(flat[offs[j]:offs[j + 1]])
                    if row.size >= 16:
                        yield f"{iid[i]}#c{c}", f, row, ts0
        return

    # 慢路径（非常见列型：嵌套结构体等）—— 与旧行为一致，不静默丢
    tgt = col.to_pylist()
    for t, f, i, st in zip(tgt, freq, iid, st_col):
        if t is None:
            continue
        a = np.asarray(t, dtype=np.float32)
        if a.ndim == 2:
            for c in range(a.shape[0]):
                row = a[c]
                # 保留 NaN 原值（掩码由适配器派生）
                if row.size >= 16:
                    yield f"{i}#c{c}", str(f), row, _ts_epoch(st)
        else:
            # 保留 NaN 原值（掩码由适配器派生）
            if a.size < 16:
                continue
            yield str(i), str(f), a, _ts_epoch(st)


def infer_freq(ts_s: np.ndarray) -> str:
    """从时间戳（秒）的中位步长推断频率标签（对齐我们既有的频率词表）。"""
    return infer_freq_int_ts(ts_s, 1.0)


_FREQ_TAB = [(1, "S"), (60, "T"), (600, "10T"), (900, "15T"), (1800, "30T"), (3600, "H"),
             (21600, "6H"), (86400, "D"), (604800, "W"), (2592000, "M"),
             (7776000, "Q-DEC"), (31536000, "A-DEC")]


def _freq_from_median(med: float) -> str:
    """中位步长（秒）→ 频率标签（对齐我们既有的频率词表 ✓）。"""
    if not np.isfinite(med) or med <= 0:
        return "?"
    best = min(_FREQ_TAB, key=lambda kv: abs(np.log(med) - np.log(kv[0])))
    return best[1] if abs(np.log(med) - np.log(best[0])) < 0.2 else "?"


def infer_freq_int_ts(vals_int: np.ndarray, div: float, max_diffs: int = 200_000) -> str:
    """整数时间戳数组（`div` = 每秒对应的整数步数）→ 频率标签。

    ★ 超长序列按等距抽样算中位步长（2026-09-13 性能修复）：weatherbench_hourly 每条
      35 万点，全量 diff 只为定一个 freq 标签不划算；等间隔序列抽样后标签不变 ✓。
    """
    if vals_int.size < 3:
        return "?"
    # ★ 抽样必须在 **diff 之后**（2026-09-13 实测踩坑）：抽点再 diff 会把步长乘 2
    #   （weatherbench 350k 点小时序 → 抽到 72min 步长 → 频率误判成 "?" ✗）。
    d = np.diff(vals_int.astype(np.float64, copy=False))
    d = d[d > 0]
    if d.size == 0:
        return "?"
    if d.size > max_diffs:
        step = int(np.ceil(d.size / max_diffs))
        d = d[::step]
    return _freq_from_median(float(np.median(d)) / div)


def _as_f32(a: np.ndarray) -> np.ndarray:
    """数值数组 → float32（已是 float32 时不复制 ✓，保留 NaN 原值 ✓）。"""
    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)


def _valid_mask(col):
    """arrow 列的有效位掩码（numpy bool）—— 只有存在 null 时才调用 ✓。"""
    import pyarrow.compute as pc
    return pc.is_valid(col).to_numpy(zero_copy_only=False)


def _flat_series_view(col):
    """「每行一条序列」的 arrow 列 → `(flat, offsets, n_chan)` 零拷贝视图。

    支持两种列型（HF 语料实测）：
      `list<数值>`                     → n_chan=1（每行一条序列）
      `fixed_size_list<list<数值>>[C]` → n_chan=C（每行展开成 C 条序列）
    其余列型返回 None（调用方退回 to_pylist 慢路径 ✓）。
    """
    import pyarrow as pa
    typ = col.type
    n_chan = 1
    if pa.types.is_fixed_size_list(typ):
        inner_t = typ.value_type
        if not (pa.types.is_list(inner_t) or pa.types.is_large_list(inner_t)):
            return None
        n_chan = int(typ.list_size)
        inner = col.flatten()          # 长度 = n_rows * C 的 list<数值>（null 位与父级对齐 ✓）
    elif pa.types.is_list(typ) or pa.types.is_large_list(typ):
        inner = col
    else:
        return None
    if not (pa.types.is_list(inner.type) or pa.types.is_large_list(inner.type)):
        return None
    vt = inner.type.value_type
    if not (pa.types.is_integer(vt) or pa.types.is_floating(vt)):
        return None
    vals = inner.values            # 子缓冲（offsets 是它的绝对下标 ✓）
    if not (pa.types.is_integer(vals.type) or pa.types.is_floating(vals.type)):
        return None
    offs = inner.offsets.to_numpy(zero_copy_only=False)
    flat = vals.to_numpy(zero_copy_only=False)
    if offs.size == 0 or flat.size == 0:
        return None
    return flat, offs, n_chan


def _ts_list_view(col):
    """`list<timestamp>` 列 → `(epoch 整数数组, offsets, 秒换算除数)` 零拷贝视图。

    只供频率推断与起始时间戳使用；不支持时返回 None（调用方退回 to_pylist ✓）。
    """
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
        ints = npv.view("int64")       # datetime64[unit] → 同一缓冲的整数视图 ✓
        div = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[np.datetime_data(npv.dtype)[0]]
    else:
        ints = col.values.cast(pa.int64()).to_numpy(zero_copy_only=False)
    if ints.size == 0:
        return None
    return ints, col.offsets.to_numpy(zero_copy_only=False), div


def _emit_column(t, val_c: str, id_c, ts_c, path: str):
    """把一个数值列按"数组列 / 长表"两种形态产出为序列。"""
    import pyarrow.parquet as pq  # noqa: F401  (保持依赖显式)
    multi = val_c != "target"                       # 非 target 列：id 带列名后缀以便区分
    tag = "" if not multi else f"@{val_c}"
    col = t.column(val_c)
    # ★ 零拷贝快速路径（2026-09-13 二次复检加）：`list<数值>` 列不再 to_pylist() ——
    #   weatherbench_hourly 单条 35 万点、单文件 8760 万点，Python 对象化实测 ~75min/分片 ✗。
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
            # 保留 NaN 原值（掩码由适配器派生）
            if v_arr.size >= 16:
                n_out += 1
                yield f"{sid}{tag}", infer_freq(ts_arr), v_arr, int(ts_arr[0])
    if n_out == 0:
        print(f"  [warn] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
              f" 列 {val_c}: 产出 0 条（**不静默**，请检查该列类型/长度）", flush=True)


def _emit_list_fast(t, n_rows: int, flat: np.ndarray, offs: np.ndarray,
                    id_c, ts_c, path: str, tag: str):
    """`list<数值>` 列的零拷贝产出（每行一条序列；语义与 `_emit_column` 慢路径一致 ✓）。

    id = `{sid}{tag}#r{i}`；freq 由该行时间戳推断（<3 个记 "?"）；有限点 <16 的行丢弃；
    NaN 原值保留（掩码由适配器派生）✓；整列 0 条时告警（不静默 ✗）。
    """
    ids = t.column(id_c).to_pylist() if id_c else [str(i) for i in range(n_rows)]
    ts_col = t.column(ts_c).combine_chunks() if ts_c else None
    ts_view = _ts_list_view(ts_col) if ts_col is not None else None
    n_out = 0
    for i in range(n_rows):
        a = _as_f32(flat[offs[i]:offs[i + 1]])
        # 保留 NaN 原值；判据与慢路径一致：有限点 <16 舍弃
        if a.size == 0 or int(np.isfinite(a).sum()) < 16:
            continue
        freq, ts0 = "?", 0
        if ts_col is not None:
            # ★ ts 口径必须与旧路径一致（等价门槛实测）：旧路径是 `datetime.timestamp()`（naive
            #   时间戳按**容器本地时区**解释）→ 这里只取首元素转 Python datetime 再走同一函数
            #   （O(1)/行 ✓，全列仍是零拷贝）。直接读 int64 会差一个时区偏移（实测 8h ✗）。
            try:
                ts0 = _ts_epoch(ts_col[i][0].as_py())
            except Exception:
                ts0 = 0
        if ts_view is not None:
            tv, toff, div = ts_view
            seg = tv[toff[i]:toff[i + 1]]
            if seg.size >= 3:
                # freq 只看步长：常数时区偏移不影响中位步长（跨夏令时的极少数行不影响标签 ✓）
                freq = infer_freq_int_ts(seg, div)
        n_out += 1
        yield f"{ids[i]}{tag}#r{i}", freq, a, ts0
    if n_out == 0:
        print(f"  [warn] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
              f" 列 {tag or 'target'}: 产出 0 条（**不静默**，请检查该列类型/长度）", flush=True)


def iter_parquet_generic(path: str):
    """稳健通用读取器（覆盖 HF 语料里的**混合 schema**）。

    实测（schema 普查，chronos 1241 个 parquet）至少 3 种列组合：
      ['id','latitude','level','longitude','subset','target','timestamp']
      ['id','target','timestamp']
      ['id','im_0','target','timestamp']
    而且 `target`/`timestamp` **可能是数组列**（每行一条序列）也可能是**标量列**（长表）。

    策略：
      1) 值列优先取 `target`；否则取第一个非 id/时间/元信息列；
      2) 若值列是数组 → **每行一条序列**（freq 由该行 timestamp 数组推断）；
      3) 若是标量 → 按 `id` 分组、按时间排序 → 每组一条序列（freq 由中位步长推断）；
      4) 时间戳缺失时不推断 freq（记 "?"），但**不丢数据**。
    """
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = t.column_names
    id_c = next((c for c in ("id", "item_id", "series_id") if c in cols), None)
    ts_c = next((c for c in ("timestamp", "start", "time", "date", "ds") if c in cols), None)
    # ★ 值列选择（2026-09-13 修复独立核对发现的静默丢数据 bug）：
    #   旧逻辑"没有 target 就取第一个非 id/时间列" —— 若那列是**字符串**（ushcn_daily 的
    #   `state`、monash_rideshare 的 `source_location`），每行转换都失败并被 except 静默吞掉
    #   → **整个数据集 0 条且不报错**（已实测丢掉 ushcn_daily 1.99 亿点）。
    #   新逻辑：只在**数值/数值数组列**里选；没有 target 时**所有数值列都产出**（每变量一条序列），
    #   但排除预报协变量（fcst_*/forecast_*/pred*，避免把预报当观测）；任何 0 条都告警。
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
        print(f"  [warn] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}: "
              f"无数值列可用（cols={cols}）→ 跳过（**不静默**）", flush=True)
        return
    t = t.select([c for c in (id_c, ts_c, *val_cols) if c and c in cols])
    val_c = val_cols[0]
    # 逐个数值列产出（单列时行为与旧版一致；多列时每列一条序列，id 带列名后缀）
    for _vc in val_cols:
        yield from _emit_column(t, _vc, id_c, ts_c, path)



def build_corpus(src: str, out: str, shard_gb: float = 8.0, max_files: int = 0,
                 dry: bool = False, part: int = 0, parts: int = 1,
                 balance: str = "index", file_mod: int = 0,
                 file_mod_n: int = 1) -> dict:
    """把 `src` 下的 arrow/parquet 转成 fast 格式，返回汇总 dict ✓。

    参数全部来自配置 ✓（本文件没有 CLI ✗，正式路径 = main.py → Pipeline →
    `pipeline/build_corpus.py::run(config)` ✓）：
      src       原始 HF 数据集根目录（首层=数据集名）
      out       输出根目录
      shard_gb  单个分片的目标大小（GB，默认 8）
      max_files 每个数据集最多读几个文件（0=不限；冒烟用）
      dry       只统计不写盘
      part/parts 并行切分：数据集按 index % parts == part 分给多个进程
                （转换瓶颈是 parquet 解码 + 逐行迭代，单进程只吃 1 核；
                 本机 100 核 → 必须并行，否则 2.8TB 要跑几十小时）
      balance   切分方式：index=按数据集编号均分（旧行为）；
                bytes=按字节数做 LPT 贪心装箱（最大的先放、每次放进最空的桶）
                → 大集不再扎堆，总完成时间由"最大数据集"而非"最重 part"决定
                （2026-09-13 实测：weatherbench_hourly 726GB 扎堆教训）
      file_mod / file_mod_n
                **数据集内**并行（2026-09-13 二次复检加）：只读该数据集文件列表里
                `i % file_mod_n == file_mod` 的那些文件 → 同一个超大集可以切成 N 份
                并行跑（各自写自己的 out 目录，最后用 merge_corpus_parts.py 合 ✓）。
                用途：weatherbench_hourly（726GB / ~900 个 parquet）单进程要 1 小时+
                且只能吃 1 核 ✗；切成 8 份后各写 `p0_split<k>/` ✓。
                注意：切分只影响「读哪些文件」，数据集清单（→ ds_id）不变 ✓。
    """

    os.makedirs(out, exist_ok=True)
    # ★ 排除隐藏目录（HF 下载缓存 .cache 会被误当数据集，实测产出 13 万个
    #   只有 1 条 0 点序列的垃圾分片）以及不含 arrow/parquet 的目录。
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
            print(f"  [skip] {_d}: 无 arrow/parquet 文件", flush=True)

    if balance == "bytes" and parts > 1:
        # LPT 贪心装箱：最大的数据集先放，每次放进当前最空的桶。
        # 目的：`weatherbench_hourly`(726GB) 这类大集不再和别的大集扎堆在同一 part，
        # 总完成时间由"最大数据集"而非"最重 part"决定（2026-09-13 实测教训）。
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
        print("  [balance=bytes] 各 part 预估: "
              + ", ".join(f"p{i}={v/1e9:.1f}G" for i, v in enumerate(load)), flush=True)
        datasets = buckets[part]
    elif parts > 1:
        datasets = [d for i, d in enumerate(all_ds) if i % parts == part]
    else:
        datasets = all_ds
    print(f"  数据集 {len(datasets)}/{len(all_ds)} 个（part {part}/{parts}）；"
          f"目标分片 {shard_gb} GB")

    limit = int(shard_gb * (1024 ** 3) / 4)          # float32 → 每分片的最大点数
    shard_idx = 0
    buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
    meta_rows = []                                     # 全局索引：(shard, local_i, len, freq, ds_id)
    ds_ids = {d: i for i, d in enumerate(datasets)}
    # ★ 数据集归属（2026-09-13 二次复检）：把「局部 ds_id → 数据集名」当场落盘。
    #   背景：`ds_id` 是 part 内局部编号，旧统计靠 `datasets.txt` 猜 → 混轮次时名字错位 ✗。
    #   src/part 也一并记下：后续可判断「某个数据集进了哪个 part」而不必重扫源目录 ✓。
    with open(f"{out}/ds_names.json", "w") as _fh:
        json.dump({"part": part, "parts": parts, "src": src,
                   "datasets": {str(i): d for i, d in enumerate(datasets)}},
                  _fh, ensure_ascii=False, indent=1)
    freq_ids: dict[str, int] = {}         # 确定性映射（禁用 hash()：受 PYTHONHASHSEED 影响，跨进程不一致）
    written_pts = 0
    written_rows = 0
    cur_pts = 0                    # 当前分片累计点数（避免每行 sum(buf_len) 的 O(n^2)）
    t0 = time.time()

    def flush():
        nonlocal shard_idx, buf_rows, buf_off, buf_len, buf_freq, buf_ds, written_pts, written_rows, cur_pts, buf_ts
        if not buf_len:
            return
        # ★ 长度守卫（2026-09-13 修）：arrow 路径缺 <16 过滤，空/超短序列被当数据落盘，
        #   实测产出了 ~9,800 个「1 条 / 0.00B 点」垃圾分片。这里统一在落盘前过滤，
        #   并保证过滤后 offsets/lengths/数据区三者自洽（offsets 重建）。
        # 长度守卫按「有限点」计（保留 NaN 后，长度不能代表有效信息量）
        _fin = [int(np.isfinite(a).sum()) for a in buf_rows]
        if any(f < 12 for f in _fin):
            keep = [i for i, f in enumerate(_fin) if f >= 12]
            dropped = len(buf_len) - len(keep)
            if dropped:
                print(f"    [长度守卫] 本分片丢弃 {dropped} 条过短(<12)序列", flush=True)
            if not keep:
                buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
                cur_pts = 0
                return
            buf_rows = [buf_rows[i] for i in keep]
            buf_len = [buf_len[i] for i in keep]
            buf_freq = [buf_freq[i] for i in keep]
            buf_ds = [buf_ds[i] for i in keep]
            # ★ buf_ts 必须一起过滤（2026-09-13 二次复检抓到：漏了它 → ts.npy 比 lengths
            #   长，抛短行之后的每一行日历时间都被错位一位 ✗；已受影响分片见
            #   script/corpus/repair_shard_ts.py 的修复记录 ✓）。
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
        print(f"    shard{shard_idx:04d}: {len(buf_len):>7d} 条 / {vals.size/1e9:5.2f}B 点 / "
              f"{vals.nbytes/1e9:5.2f} GB  [{time.time()-t0:.0f}s]", flush=True)
        meta_rows.append((shard_idx, len(buf_len), int(vals.size)))
        shard_idx += 1
        buf_rows, buf_off, buf_len, buf_freq, buf_ds, buf_ts = [], [0], [], [], [], []
        cur_pts = 0        # ★ 必须与缓冲一起重置（2026-09-13 修）：漏了它 → 越阈值后
                           #   每行都 flush → 产出上万个「1 条」碎片分片

    n_files = 0
    for ds in datasets:
        # ★ 格式自动识别：GiftEvalPretrain 是 .arrow 宽表；Chronos/autogluon 是 .parquet 长表
        #   （实测：dominick 的列是 id/timestamp/consumption_kW —— 与宽表完全不同，
        #    不识别就会静默产出 0 条，我们已踩过一次）
        fs_arrow = sorted(glob.glob(os.path.join(src, ds, "**", "*.arrow"), recursive=True))
        fs_parq = sorted(glob.glob(os.path.join(src, ds, "**", "*.parquet"), recursive=True))
        # ★ 排除 HF 流式缓存副本（2026-09-13 独立核对发现的重复源）：
        #   数据集目录内可能残留 cache-<hash>.arrow（内容与 data-*.arrow 相同），
        #   被一并摄入 → 序列重复 2-3 倍（实测 cmip6_1850 3×、pdb/spain/gfc12_load/gfc17_load 2×）。
        _before = len(fs_arrow) + len(fs_parq)
        fs_arrow = [f for f in fs_arrow if not os.path.basename(f).startswith("cache-")]
        fs_parq = [f for f in fs_parq if not os.path.basename(f).startswith("cache-")]
        _dropped = _before - len(fs_arrow) - len(fs_parq)
        if _dropped:
            print(f"    [去重] {ds}: 排除 {_dropped} 个 HF 缓存副本文件（cache-*.arrow）", flush=True)
        if fs_parq and not fs_arrow:
            files = fs_parq
            reader_fn = iter_parquet_generic
        else:
            files = fs_arrow
            reader_fn = iter_arrow_series
        if file_mod_n > 1:
            # 数据集内并行：只取第 file_mod 份文件（各份写各自的 out 目录 ✓）
            _n_all = len(files)
            files = [f for i, f in enumerate(files) if i % file_mod_n == file_mod]
            print(f"    [file_mod] {ds}: 第 {file_mod}/{file_mod_n} 份 → "
                  f"{len(files)}/{_n_all} 个文件", flush=True)
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
                # ★ 必须**早 flush**：原判据要等 200 万行/21 亿点才落盘 →
                #   首个分片迟迟不出现、内存里囤 8GB 数组 + 上百万 Python 对象。
                #   改为行数上限 5 万 ⇒ 稳定出分片、内存可控、进度可见。
                if len(buf_off) - 1 >= 50_000 or cur_pts >= limit:
                    if not dry:
                        flush()
            if max_files and n_files >= max_files:
                break
        if n_files % 50 == 0:
            print(f"    ... 已处理 {n_files} 个文件 / {written_rows:,} 条", flush=True)
    if not dry:
        flush()

    # 全局元数据 + 长度分桶（只建索引，不动数据排布）
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
                    "notes": "values=float32 连续；offsets 给出 O(1) 取行；无长度/上下文截断"}
        with open(f"{out}/manifest.json", "w") as _fh:
            json.dump(manifest, _fh, indent=1)
        print(f"  ✓ 完成：{shard_idx} 分片 / {L.size:,} 序列 / {L.sum()/1e9:.2f}B 点 / "
              f"{L.sum()*4/1e9:.1f} GB  [{time.time()-t0:.0f}s]")
        summary = {"out": out, "n_shards": shard_idx, "n_rows": int(L.size),
                   "n_points": int(L.sum()), "n_datasets": len(datasets),
                   "dry": False}
    else:
        print(f"  [dry] 将写入 {written_rows:,} 序列 / {written_pts/1e9:.2f}B 点 / "
              f"{written_pts*4/1e9:.1f} GB")
        summary = {"out": out, "n_shards": 0, "n_rows": int(written_rows),
                   "n_points": int(written_pts), "n_datasets": len(datasets),
                   "dry": True}
    return summary
