from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepSpan:
    label: str
    start_time: float
    duration_s: float
    error: bool = False
    input_shape: tuple | None = None
    output_shape: tuple | None = None
    child_trace: InvocationTrace | None = None
    output_value: Any = None
    operator_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvocationTrace:
    spans: list[StepSpan] = field(default_factory=list)
    total_duration_s: float = 0.0
    batch_size: float | None = None
    workers: int | None = None

    def span_fractions(self) -> dict[str, float]:
        if self.total_duration_s == 0.0:
            return {s.label: 0.0 for s in self.spans}
        return {s.label: s.duration_s / self.total_duration_s for s in self.spans}


@dataclass
class TracingConfig:
    collector: TraceCollector
    operator_labels: list[str] | None = None
    capture_shapes: bool = False
    capture_values: bool = False


class TraceCollector(ABC):
    @abstractmethod
    def on_trace(self, trace: InvocationTrace) -> None: ...


class _NoOpSpanList:
    """Accepts appends and discards them — used by _NoOpTrace."""
    def append(self, _: Any) -> None:
        pass


class _NoOpTrace:
    """Stands in for InvocationTrace when no collector is attached.

    All mutations are accepted and silently discarded so _step and
    _step_into_batch need no if-guards — there is one code path.
    """
    def __init__(self, batch_size: int | None = None) -> None:
        self.spans: Any = _NoOpSpanList()
        self.total_duration_s: float = 0.0
        self.batch_size = batch_size


def merge_traces(traces: list[InvocationTrace]) -> InvocationTrace:
    """Return a new InvocationTrace whose per-span durations are the mean across *traces*."""
    if not traces:
        return InvocationTrace()
    n = len(traces)
    traces_with_batch_size = [t for t in traces if t.batch_size is not None]
    # Collect all span labels in first-seen order.
    seen: dict[str, list[StepSpan]] = {}
    for t in traces:
        for s in t.spans:
            seen.setdefault(s.label, []).append(s)
    spans = [
        StepSpan(
            label=label,
            start_time=0.0,
            duration_s=sum(s.duration_s for s in group) / n,
            child_trace=merge_traces([s.child_trace for s in group if s.child_trace is not None]) if any(s.child_trace for s in group) else None,
        )
        for label, group in seen.items()
    ]
    return InvocationTrace(
        spans=spans,
        total_duration_s=sum(t.total_duration_s for t in traces) / n,
        batch_size=sum(t.batch_size for t in traces_with_batch_size) / len(traces_with_batch_size)
        if traces_with_batch_size else None,
        workers=traces[0].workers,
    )


def _extract_shape(value: Any) -> tuple | None:
    if hasattr(value, "array") and hasattr(value.array, "shape"):
        return tuple(value.array.shape)
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None
