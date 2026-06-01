from __future__ import annotations

from ..tracing import InvocationTrace, accumulate_trace_mean
from .concurrent_collector import ConcurrentCollector


class AggregateCollector(ConcurrentCollector):
    """Accumulates per-operator latency stats across multiple invocations.

    Traces are processed on a background thread — call flush() before reading
    results to ensure all in-flight traces have been incorporated.

    Call ``print_summary()`` to display stats and an average invocation trace.
    """

    def __init__(self) -> None:
        super().__init__()
        self._calls: int = 0
        self._avg_trace: InvocationTrace = InvocationTrace()

    def _collect(self, trace: InvocationTrace) -> None:
        self._calls += 1
        accumulate_trace_mean(self._avg_trace, trace, self._calls)

    @property
    def total_calls(self) -> int:
        return self._calls

    @property
    def avg_pipeline_latency_ms(self) -> float:
        return self._avg_trace.total_duration_s * 1000

    @property
    def avg_trace(self) -> InvocationTrace:
        return self._avg_trace

    def reset(self) -> None:
        self._calls = 0
        self._avg_trace = InvocationTrace()

    def print_summary(self) -> None:
        if self._calls == 0:
            print("  (no calls recorded)")
            return
        print(f"  Calls                : {self._calls}")
        print(f"  Latency Avg.         : {self.avg_pipeline_latency_ms:.2f}ms")
        print()
        print(self.avg_trace)
