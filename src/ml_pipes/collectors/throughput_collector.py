from __future__ import annotations

import time
from collections import deque

from ..tracing import InvocationTrace
from .aggregate_collector import AggregateCollector


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class ThroughputCollector(AggregateCollector):
    """Extends AggregateCollector with throughput tracking: long-term FPS (since
    start) and short-term FPS (last ``window_s`` seconds) to expose throttles
    and chokes.

    The live status line overwrites itself every ``report_interval_s`` seconds.
    Pass ``target_fps`` to add a coverage percentage to the status line.
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

    def _collect(self, trace: InvocationTrace) -> None:
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
            self._t_last_report = now
        self._window.append(now)
        self._latency_window.append(trace.total_duration_s)
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

        print(f"\r  {fps_str} / latency: {latency:.1f}ms", end="", flush=True)

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
        self._min_fps = float("inf")
        self._max_fps = 0.0

    def print_summary(self) -> None:
        print()  # move past the last \r line
        elapsed = time.perf_counter() - self._t_start if self._t_start else 0.0
        fps = self.fps
        min_fps = self._min_fps if self._min_fps != float("inf") else 0.0
        if self.target_fps is not None:
            coverage = min(fps / self.target_fps * 100, 100.0)
            fps_str = f"{fps:.1f} / {self.target_fps:.1f} target ({coverage:.0f}%)"
        else:
            fps_str = f"{fps:.1f}"
        print(f"  Frames               : {self._calls}")
        print(f"  FPS Avg.             : {fps_str}")
        print(f"  FPS min / max        : {min_fps:.1f} / {self._max_fps:.1f}")
        print(f"  Latency Avg.         : {self.avg_pipeline_latency_ms:.2f}ms")
        print()
        fracs = self.operator_fractions()
        avgs = self.avg_operator_latency_ms()
        print(f"  {'Operator':<35} {'Avg ms':>8}  {'% of total':>10}")
        print(f"  {'-' * 35} {'-' * 8}  {'-' * 10}")
        for label in self._op_total:
            print(f"  {label:<35} {avgs[label]:>8.2f}ms {fracs[label] * 100:>9.1f}%")
