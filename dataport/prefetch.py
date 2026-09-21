"""分片预读：把「逐行小随机读」变成「顺序大块读」，专门治 Lustre 冷读 ✗。

## 为什么需要它（2026-09-14 实测，`script/diagnostics/io_probe.py`）

| 读法 | 冷（不在页缓存） | 热 |
| --- | --- | --- |
| 顺序大块读（8 MB 块） | ~1.1 GB/s | ~4.3 GB/s |
| 按 offsets **逐行读**（dataloader 的真实读法） | **3.4–4.0 ms/行** | 0.01–0.08 ms/行 |

生产（16 任务 × 16 worker、4 卡 × 4 任务）实测 loader 卡在 ~4.4k 样本/s/任务 ≈ 3.6 ms/样本，
与「冷读逐行」的 3.5 ms/样本**完全吻合** → 瓶颈是**冷读 IO**，不是 CPU 预处理 ✓。
（语料 2,053 GB > 内存 1,868 GB，所以页缓存永远装不下全量 ✗。）

## 做法

每个训练进程一个**后台预读线程**：sampler 进入某个分片时，把**后面 lookahead 个分片**
按 8 MB 块顺序读一遍（只为了灌页缓存，数据丢掉 ✓）。读的带宽 1.1 GB/s ≫ 逐行的 ~0.3 GB/s，
而且整块顺序读让 Lustre 走大 RPC（逐行读是 30 KB 级别的小 RPC ✗）。

## 进程间去重（关键，不然 16 个任务会把带宽打爆）

16 个训练任务的 sampler 顺序**完全一样**（同 seed ✓），所以它们想要的是同一批分片。
去重用 `tmp/prefetch/<sha1(路径)>.stamp`：

- 抢占：`O_CREAT|O_EXCL` 建 stamp，建成功才读（谁先到谁读 ✓）；
- 别人的 stamp 新鲜（TTL 内）→ 直接跳过（他读的就是我要读的 ✓）；
- stamp 过期 → 说明上一个读的人挂了/读完太久，重新读一遍 ✓。

## 边界

- 预读**不碰数据**：只读页缓存，不改任何样本 → 不影响等价性门槛 ✓；
- 预读线程是 daemon，队列有上限（读不过来就丢任务，不积压内存 ✗）；
- `FT_PREFETCH=0` 可整体关掉（做 A/B 用 ✓）。
"""
from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _default_stamp_dir() -> Path:
    """Stamp directory kept inside the checkout and ignored by git."""
    env = os.environ.get("FT_PREFETCH_DIR")
    if env:
        return Path(env)
    return _REPO / "tmp" / "prefetch"


class ShardPrefetcher:
    """后台顺序预读分片（见模块 docstring ✓）。"""

    def __init__(self, lookahead: int = 2, chunk_mb: int = 8,
                 ttl_s: float = 900.0, stamp_dir: Path | None = None,
                 enabled: bool | None = None):
        self.lookahead = max(1, int(lookahead))
        self.chunk = max(1, int(chunk_mb)) << 20
        self.ttl_s = float(ttl_s)
        self.stamp_dir = Path(stamp_dir) if stamp_dir else _default_stamp_dir()
        if enabled is None:
            enabled = os.environ.get("FT_PREFETCH", "1") != "0"
        self.enabled = bool(enabled)
        self.stats = {"queued": 0, "read": 0, "skipped": 0, "bytes": 0, "fail": 0}
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue(
            maxsize=max(4, self.lookahead * 4))
        self._seen: set[str] = set()          # 本进程已入队的路径（不重复入队 ✓）
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if self.enabled:
            try:
                self.stamp_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False          # 建不出目录就安静退化（不影响训练 ✓）
        self._buf = bytearray(self.chunk)

    # ---- 主线程接口 ----
    def warm(self, paths: list[str | Path]) -> None:
        """把接下来要用的分片路径交给后台线程（不阻塞 ✓）。"""
        if not self.enabled:
            return
        for p in paths:
            sp = str(p)
            if sp in self._seen:
                continue
            self._seen.add(sp)
            try:
                self._q.put_nowait((sp, self._stamp_path(sp)))
            except queue.Full:
                self._seen.discard(sp)        # 队列满了：等下次再提（不阻塞 ✗）
                return
            self.stats["queued"] += 1
        self._ensure_thread()

    def close(self, wait: bool = False) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and wait:
            th.join(timeout=5.0)

    # ---- 后台线程 ----
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="shard-prefetch",
                                            daemon=True)
            self._thread.start()

    def _stamp_path(self, path: str) -> str:
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        return str(self.stamp_dir / f"{Path(path).name}.{h}.stamp")

    def _claim(self, stamp: str) -> bool:
        """抢占该分片的预读权：新鲜 stamp 存在 → 别人在管/刚管完 → 跳过 ✓。"""
        try:
            fd = os.open(stamp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                fresh = (time.time() - os.stat(stamp).st_mtime) < self.ttl_s
            except OSError:
                fresh = False
            if fresh:
                self.stats["skipped"] += 1
                return False
            try:
                os.utime(stamp, None)         # 过期 → 抢占（刷 mtime ✓）
            except OSError:
                pass
            return True
        except OSError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def _read_seq(self, path: str) -> None:
        with open(path, "rb", buffering=0) as fh:
            while True:
                if self._stop.is_set():
                    return
                n = fh.readinto(self._buf)
                if not n:
                    break
                self.stats["bytes"] += n
        self.stats["read"] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                path, stamp = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self._claim(stamp):
                continue
            t0 = time.perf_counter()
            try:
                self._read_seq(path)
            except Exception:                 # noqa: BLE001（预读失败绝不影响训练 ✓）
                self.stats["fail"] += 1
            else:
                self.stats["last_s"] = round(time.perf_counter() - t0, 2)
