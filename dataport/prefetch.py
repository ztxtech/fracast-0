"""Background shard prefetching for storage-bound training jobs.

The training reader performs offset-based random row access.  On a cold file
system, those small reads can be much slower than sequential block reads.  A
single background thread reads upcoming shards in large blocks to warm the page
cache without changing the data returned to the sampler.
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
    """Warm the page cache by reading upcoming shards sequentially."""

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
        self._seen: set[str] = set()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if self.enabled:
            try:
                self.stamp_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False
        self._buf = bytearray(self.chunk)

    # Public interface used by the sampler thread.
    def warm(self, paths: list[str | Path]) -> None:
        """Queue upcoming shard paths without blocking the caller."""
        if not self.enabled:
            return
        for path in paths:
            key = str(path)
            if key in self._seen:
                continue
            self._seen.add(key)
            try:
                self._q.put_nowait((key, self._stamp_path(key)))
            except queue.Full:
                self._seen.discard(key)
                return
            self.stats["queued"] += 1
        self._ensure_thread()

    def close(self, wait: bool = False) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and wait:
            th.join(timeout=5.0)

    # Background worker implementation.
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
        """Claim prefetch ownership for one shard."""
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
                os.utime(stamp, None)
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
            except Exception:  # noqa: BLE001 - prefetch must never stop training
                self.stats["fail"] += 1
            else:
                self.stats["last_s"] = round(time.perf_counter() - t0, 2)
