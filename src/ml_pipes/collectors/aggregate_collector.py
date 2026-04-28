from __future__ import annotations

from ..tracing import InvocationTrace, StepSpan
from .concurrent_collector import ConcurrentCollector
from .print_collector import PrintCollector


def _aggregate_traces(avg: InvocationTrace, incoming: InvocationTrace, n: int) -> None:
    """Update *avg* in-place with a new *incoming* trace using an incremental mean.

    *n* is the new total call count (already incremented by the caller) so that
    avg = prev_avg + (incoming - prev_avg) / n.
    """
    avg.total_duration_s += (incoming.total_duration_s - avg.total_duration_s) / n

    incoming_by_label = {s.label: s for s in incoming.spans}

    for span in avg.spans:
        inc = incoming_by_label.get(span.label)
        if inc is None:
            continue
        span.duration_s += (inc.duration_s - span.duration_s) / n
        if span.child_trace is not None and inc.child_trace is not None:
            span.child_trace.batch_size = inc.child_trace.batch_size
            span.child_trace.workers = inc.child_trace.workers
            _aggregate_traces(span.child_trace, inc.child_trace, n)

    existing_labels = {s.label for s in avg.spans}
    for label, inc in incoming_by_label.items():
        if label not in existing_labels:
            child = (
                InvocationTrace(
                    batch_size=inc.child_trace.batch_size,
                    workers=inc.child_trace.workers,
                )
                if inc.child_trace is not None else None
            )
            avg.spans.append(StepSpan(
                label=label,
                start_time=0.0,
                duration_s=inc.duration_s,
                child_trace=child,
            ))
            if child is not None:
                _aggregate_traces(child, inc.child_trace, 1)


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
        self._printer: PrintCollector = PrintCollector()

    def _collect(self, trace: InvocationTrace) -> None:
        self._calls += 1
        _aggregate_traces(self._avg_trace, trace, self._calls)

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
        self._printer.print_trace(self.avg_trace)
