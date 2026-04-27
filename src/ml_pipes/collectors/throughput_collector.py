from __future__ import annotations

import os
import time
from collections import deque

from ..tracing import InvocationTrace
from .aggregate_collector import AggregateCollector

try:
    import psutil as _psutil
    _proc = _psutil.Process(os.getpid())
    _cpu_count = _psutil.cpu_count(logical=True) or 1
    _total_mem = _psutil.virtual_memory().total
except ImportError:
    _proc = None
    _cpu_count = 1
    _total_mem = 0


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _fmt_mem(bytes_: int) -> str:
    if bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.0f}KB"
    if bytes_ < 1024 ** 3:
        return f"{bytes_ / 1024 ** 2:.0f}MB"
    return f"{bytes_ / 1024 ** 3:.1f}GB"


class ThroughputCollector(AggregateCollector):
    """Extends AggregateCollector with throughput tracking: long-term FPS (since
    start) and short-term FPS (last ``window_s`` seconds) to expose throttles
    and chokes.

    The live status line overwrites itself every ``report_interval_s`` seconds.
    Pass ``target_fps`` to add a coverage percentage to the status line.
    If ``psutil`` is installed, CPU% and RSS memory are also reported.
    """

    def __init__(
        self,
        target_fps: float | None = None,
        report_interval_s: float = 1.0,
        window_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_fps = target_fps
        self._report_interval_s = report_interval_s
        self._window_s = window_s
        self._t_start: float | None = None
        self._t_last_report: float | None = None
        self._window: deque[float] = deque()
        self._latency_window: deque[float] = deque()
        self._min_fps: float = float("inf")
        self._max_fps: float = 0.0
        self._cpu_window: deque[float] = deque()
        self._cpu_total: float = 0.0
        self._cpu_samples: int = 0
        # Prime psutil so the first non-blocking call returns a valid value
        if _proc is not None:
            _proc.cpu_percent(interval=None)

    def _collect(self, trace: InvocationTrace) -> None:
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
            self._t_last_report = now
        self._window.append(now)
        self._latency_window.append(trace.total_duration_s)
        if _proc is not None:
            cpu = _proc.cpu_percent(interval=None) / _cpu_count
            self._cpu_window.append(cpu)
            self._cpu_total += cpu
            self._cpu_samples += 1
        self._evict(now)
        super()._collect(trace)
        if now - self._t_last_report >= self._report_interval_s:
            self._t_last_report = now
            short = self.window_fps
            if short > 0:
                self._min_fps = min(self._min_fps, short)
                self._max_fps = max(self._max_fps, short)
            self._print_fps_line()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._window and self._window[0] < cutoff:
            self._window.popleft()
            self._latency_window.popleft()
            if self._cpu_window:
                self._cpu_window.popleft()

    def _print_fps_line(self) -> None:
        short = self.window_fps
        long_ = self.fps
        elapsed = time.perf_counter() - self._t_start if self._t_start else 0.0
        short_label = f"{self._window_s:.1f}s"
        long_label = _fmt_elapsed(elapsed)
        latency = (sum(self._latency_window) / len(self._latency_window) * 1000) if self._latency_window else 0.0

        if self.target_fps is not None:
            coverage_short = min(short / self.target_fps * 100, 100.0)
            coverage_long = min(long_ / self.target_fps * 100, 100.0)
            t = self.target_fps
            fps_str = (
                f"FPS[{short_label}]: {short:.1f} / {t:.1f} ({coverage_short:.0f}%)"
                f" / FPS[{long_label}]: {long_:.1f} / {t:.1f} ({coverage_long:.0f}%)"
            )
        else:
            fps_str = f"FPS[{short_label}]: {short:.1f} / FPS[{long_label}]: {long_:.1f}"

        resource_str = ""
        if _proc is not None:
            cpu = sum(self._cpu_window) / len(self._cpu_window) if self._cpu_window else 0.0
            mem = _proc.memory_info().rss
            mem_pct = mem / _total_mem * 100 if _total_mem else 0.0
            resource_str = f" / CPU: {cpu:.0f}% MEM: {_fmt_mem(mem)} ({mem_pct:.0f}%)"

        print(f"\r  {fps_str} / latency: {latency:.1f}ms{resource_str}", end="", flush=True)

    @property
    def fps(self) -> float:
        """Long-term FPS since the first invocation."""
        if self._calls == 0 or self._t_start is None:
            return 0.0
        return self._calls / (time.perf_counter() - self._t_start)

    @property
    def window_fps(self) -> float:
        """Short-term FPS over the last ``window_s`` seconds."""
        if not self._window:
            return 0.0
        return len(self._window) / self._window_s

    def reset(self) -> None:
        super().reset()
        self._t_start = None
        self._t_last_report = None
        self._window.clear()
        self._latency_window.clear()
        self._cpu_window.clear()
        self._cpu_total = 0.0
        self._cpu_samples = 0
        self._min_fps = float("inf")
        self._max_fps = 0.0

    def print_summary(self) -> None:
        print()  # move past the last \r line
        if _proc is not None:
            cpu_avg = self._cpu_total / self._cpu_samples if self._cpu_samples else 0.0
            mem = _proc.memory_info().rss
            mem_pct = mem / _total_mem * 100 if _total_mem else 0.0
            print(f"  CPU Avg.             : {cpu_avg:.0f}%")
            print(f"  Memory (RSS)         : {_fmt_mem(mem)} ({mem_pct:.0f}% of system)")
        fps = self.fps
        min_fps = self._min_fps if self._min_fps != float("inf") else 0.0
        if self.target_fps is not None:
            coverage = min(fps / self.target_fps * 100, 100.0)
            fps_str = f"{fps:.1f} / {self.target_fps:.1f} target ({coverage:.0f}%)"
        else:
            fps_str = f"{fps:.1f}"
        print(f"  FPS Avg.             : {fps_str}")
        print(f"  FPS min / max        : {min_fps:.1f} / {self._max_fps:.1f}")
        super().print_summary()
