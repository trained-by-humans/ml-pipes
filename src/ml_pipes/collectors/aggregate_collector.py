from __future__ import annotations

from ..tracing import InvocationTrace, TraceCollector


class AggregateCollector(TraceCollector):
    """Accumulates per-operator latency stats across multiple invocations.

    Call ``print_summary()`` to display average ms and % of total per operator.
    """

    def __init__(self) -> None:
        self._calls: int = 0
        self._total_s: float = 0.0
        self._op_total: dict[str, float] = {}
        self._op_calls: dict[str, int] = {}

    def on_trace(self, trace: InvocationTrace) -> None:
        self._calls += 1
        self._total_s += trace.total_duration_s
        for span in trace.spans:
            self._op_total[span.label] = self._op_total.get(span.label, 0.0) + span.duration_s
            self._op_calls[span.label] = self._op_calls.get(span.label, 0) + 1

    @property
    def total_calls(self) -> int:
        return self._calls

    @property
    def avg_pipeline_latency_ms(self) -> float:
        if self._calls == 0:
            return 0.0
        return self._total_s / self._calls * 1000

    def avg_operator_latency_ms(self) -> dict[str, float]:
        return {
            label: self._op_total[label] / self._op_calls[label] * 1000
            for label in self._op_total
        }

    def operator_fractions(self) -> dict[str, float]:
        """Each operator's average latency as a fraction of average pipeline latency."""
        avg_total_s = self._total_s / self._calls if self._calls else 0.0
        if avg_total_s == 0.0:
            return {label: 0.0 for label in self._op_total}
        return {
            label: (self._op_total[label] / self._op_calls[label]) / avg_total_s
            for label in self._op_total
        }

    def reset(self) -> None:
        self._calls = 0
        self._total_s = 0.0
        self._op_total.clear()
        self._op_calls.clear()

    def print_summary(self) -> None:
        if self._calls == 0:
            print("  (no calls recorded)")
            return
        print(f"  Calls : {self._calls}")
        print(f"  Avg pipeline latency : {self.avg_pipeline_latency_ms:.2f}ms")
        print()
        fracs = self.operator_fractions()
        avgs = self.avg_operator_latency_ms()
        print(f"  {'Operator':<35} {'Avg ms':>8}  {'% of total':>10}")
        print(f"  {'-' * 35} {'-' * 8}  {'-' * 10}")
        for label in self._op_total:
            print(f"  {label:<35} {avgs[label]:>8.2f}ms {fracs[label] * 100:>9.1f}%")
