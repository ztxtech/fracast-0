from __future__ import annotations

import time
from datetime import datetime


class Timer:
    """Measure elapsed wall time as a context manager or manual timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def __enter__(self) -> Timer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    @property
    def elapsed(self) -> float:
        """Return elapsed seconds."""
        return time.perf_counter() - self._start

    def reset(self) -> None:
        """Restart the timer."""
        self._start = time.perf_counter()


def now_str(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """Return a timestamp suitable for a run name."""
    return datetime.now().strftime(fmt)
