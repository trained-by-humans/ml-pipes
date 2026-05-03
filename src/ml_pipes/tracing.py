from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepSpan:
    label: str
    start_time: float
    duration_s: float
    error: bool = False
    operator_config: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple | str | None = None
    output_shape: tuple | str | None = None
    output_value: Any = None
    child_trace: InvocationTrace | None = None

    def __repr__(self) -> str:
        parts = [f"{self.label} {self.duration_s * 1000:.2f}ms"]
        if self.error:
            parts.append("!")
        if self.output_shape:
            parts.append(f"→ {self.output_shape}")
        return " ".join(parts)


def _fmt_batch_size(batch_size: float | None) -> str:
    if batch_size is None:
        return "?"
    if float(batch_size).is_integer():
        return str(int(batch_size))
    return f"{batch_size:.1f}"


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

    def __repr__(self) -> str:
        return _fmt_trace(self)


def _fmt_trace(trace: "InvocationTrace", indent: int = 0) -> str:
    prefix = "  " * indent
    fracs = trace.span_fractions()
    lines = []
    for span in trace.spans:
        mark = " !" if span.error else ""
        shape = f"  → {span.output_shape}" if span.output_shape else ""
        config = f"  cfg={span.operator_config}" if span.operator_config else ""
        label = span.label[:29] + "…" if len(span.label) > 30 else span.label
        lines.append(
            f"{prefix}  {label:30s} {span.duration_s * 1000:7.2f}ms"
            f"  ({fracs[span.label] * 100:4.1f}%){mark}{shape}{config}"
        )
        if span.child_trace is not None:
            ct = span.child_trace
            if ct.workers is not None:
                annotation = f" [n_items={_fmt_batch_size(ct.batch_size)}, concurrency={ct.workers}]"
            elif ct.batch_size is not None:
                annotation = f" [batch_size={_fmt_batch_size(ct.batch_size)}]"
            else:
                annotation = ""
            lines.append(f"{prefix}    ↳ child trace{annotation}:")
            lines.append(_fmt_trace(span.child_trace, indent + 2))
    lines.append(f"{prefix}  {'total':30s} {trace.total_duration_s * 1000:7.2f}ms")
    return "\n".join(lines)


@dataclass
class TracingConfig:
    collector: TraceCollector
    operator_labels: list[str] | None = None
    capture_config: bool = False
    capture_shapes: bool = False
    capture_outputs: bool = False


def snapshot(value: Any) -> Any:
    """Deep-copy *value* so a span captures a point-in-time snapshot."""
    return copy.deepcopy(value)


def operator_config(op: Any) -> dict[str, Any]:
    """Return public instance attributes of *op* for tooltip/tracing display."""
    return {k: v for k, v in vars(op).items() if not k.startswith("_") and k != "pipeline"}


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


def _extract_shape(value: Any) -> str | None:
    name = type(value).__name__
    # ImagePayload, TensorPayload (have .array.shape)
    if hasattr(value, "array") and hasattr(value.array, "shape"):
        return f"{name} {value.array.shape}"
    # bare numpy array
    if hasattr(value, "shape"):
        return f"{name} {value.shape}"
    # TensorRegistry: one "key: shape" entry per tensor
    if hasattr(value, "_tensors") and isinstance(getattr(value, "_tensors", None), dict):
        entries = ", ".join(f"{k}: {v.shape}" for k, v in value._tensors.items())
        return f"{name} {{{entries}}}"
    # RuntimeOutputs: named output tensors
    if hasattr(value, "names") and hasattr(value, "tensors"):
        entries = ", ".join(f"{n}: {t.array.shape}" for n, t in zip(value.names, value.tensors))
        return f"{name} {{{entries}}}"
    # Detections / Segmentations
    if hasattr(value, "boxes") and hasattr(value, "scores"):
        n = len(value.boxes)
        suffix = " + masks" if hasattr(value, "masks") else ""
        return f"{name} ({n}{suffix})"
    # bytes
    if isinstance(value, bytes):
        return f"{name} ({len(value)} B)"
    # tuple: recurse, join with " · "
    if isinstance(value, tuple):
        parts = [_extract_shape(v) or type(v).__name__ for v in value]
        return "(" + " · ".join(parts) + ")"
    if isinstance(value, list):
        return f"{name} [{len(value)}]"
    return name
