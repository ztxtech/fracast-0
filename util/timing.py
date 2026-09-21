from __future__ import annotations

import time
from datetime import datetime


class Timer:
    """计时器：可作上下文管理器，也可手动读取 elapsed。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def __enter__(self) -> Timer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    @property
    def elapsed(self) -> float:
        """返回自计时开始以来的秒数。"""
        return time.perf_counter() - self._start

    def reset(self) -> None:
        """重新开始计时。"""
        self._start = time.perf_counter()


def now_str(fmt: str = "%Y%m%d-%H%M%S") -> str:
    """返回当前时间字符串，默认适合作为 run 名称。"""
    return datetime.now().strftime(fmt)
