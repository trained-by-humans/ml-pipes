from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import GeneratorType
from typing import Any


@dataclass
class StepSpan:
    label: str
    start_time: float
    duration_s: float
    error: bool = False
    operator_config: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple | str | None = None
    output_shape: tuple | str | None = None
    output_value: Any = None
    child_trace: InvocationTrace | None = None
    operator_type: type | None = None

    def __repr__(self) -> str:
        parts = [f"{self.label} {self.duration_s * 1000:.2f}ms"]
        if self.error:
            parts.append("!")
        if self.output_shape:
            parts.append(f"→ {self.output_shape}")
        if self.attributes:
            parts.append(f"attributes={self.attributes}")
        return " ".join(parts)


@dataclass
class PendingSpan:
    """Mutable span used while lazy regions are still unresolved."""

    label: str
    start_time: float
    duration_s: float
    error: bool = False
    operator_config: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple | str | None = None
    output_shape: tuple | str | None = None
    output_value: Any = None
    child_trace: InvocationTrace | None = None
    operator_type: type | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or not getattr(self, "_closed", False):
            object.__setattr__(self, name, value)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        object.__setattr__(self, "_closed", True)

    def freeze(self) -> StepSpan:
        frozen = StepSpan(
            label=self.label,
            start_time=self.start_time,
            duration_s=self.duration_s,
            error=self.error,
            operator_config=copy.deepcopy(self.operator_config),
            attributes=copy.deepcopy(self.attributes),
            input_shape=copy.deepcopy(self.input_shape),
            output_shape=copy.deepcopy(self.output_shape),
            output_value=capture_value(self.output_value),
            child_trace=freeze_trace(self.child_trace) if self.child_trace is not None else None,
            operator_type=self.operator_type,
        )
        self.close()
        return frozen


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


def _freeze_span(span: StepSpan | PendingSpan) -> StepSpan:
    if isinstance(span, PendingSpan):
        return span.freeze()

    return StepSpan(
        label=span.label,
        start_time=span.start_time,
        duration_s=span.duration_s,
        error=span.error,
        operator_config=copy.deepcopy(span.operator_config),
        attributes=copy.deepcopy(span.attributes),
        input_shape=copy.deepcopy(span.input_shape),
        output_shape=copy.deepcopy(span.output_shape),
        output_value=capture_value(span.output_value),
        child_trace=freeze_trace(span.child_trace) if span.child_trace is not None else None,
        operator_type=span.operator_type,
    )


def freeze_trace(trace: InvocationTrace) -> InvocationTrace:
    return InvocationTrace(
        spans=[_freeze_span(span) for span in trace.spans],
        total_duration_s=trace.total_duration_s,
        batch_size=trace.batch_size,
        workers=trace.workers,
    )


def _fmt_trace(trace: "InvocationTrace", indent: int = 0) -> str:
    prefix = "  " * indent
    fracs = trace.span_fractions()
    lines = []
    for span in trace.spans:
        mark = " !" if span.error else ""
        shape = f"  → {span.output_shape}" if span.output_shape else ""
        config = f"  cfg={span.operator_config}" if span.operator_config else ""
        attributes = f"  attributes={span.attributes}" if span.attributes else ""
        label = span.label[:29] + "…" if len(span.label) > 30 else span.label
        lines.append(
            f"{prefix}  {label:30s} {span.duration_s * 1000:7.2f}ms"
            f"  ({fracs[span.label] * 100:4.1f}%){mark}{shape}{config}{attributes}"
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
    """Internal trace-capture settings used by pipeline execution.

    Collector attachment lives on Pipeline. This config only controls optional
    capture policy so internal callers such as inspect() can reuse trace
    machinery without widening the public tracing surface.
    """

    capture_config: bool = False
    capture_shapes: bool = False
    _capture_outputs: bool = False  # used internally by Pipeline.inspect(); not exposed via set_tracing()


def capture_value(value: Any) -> Any:
    """Deep-copy *value* so a span captures a point-in-time snapshot.

    When a value contains runtime-only state such as generators or iterators,
    fall back to a display-safe structural snapshot instead of failing the
    entire inspection run.
    """
    try:
        return copy.deepcopy(value)
    except Exception:
        return _capture_value_fallback(value, seen=set())


_CAPTURE_VALUE_PRIMITIVES = (bool, int, float, str, bytes, type(None))


def _capture_value_fallback(value: Any, *, seen: set[int]) -> Any:
    if isinstance(value, _CAPTURE_VALUE_PRIMITIVES):
        return value

    value_id = id(value)
    if value_id in seen:
        return "<cycle>"
    seen.add(value_id)
    try:
        if isinstance(value, (GeneratorType, Iterator)):
            return repr(value)

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            field_values = {
                field.name: _capture_value_fallback(getattr(value, field.name), seen=seen)
                for field in dataclasses.fields(value)
            }
            try:
                return type(value)(**field_values)
            except Exception:
                return field_values

        if isinstance(value, Mapping):
            snapshot_mapping = {}
            for key, item in value.items():
                snapshot_key = _capture_value_fallback(key, seen=seen)
                try:
                    hash(snapshot_key)
                except Exception:
                    snapshot_key = repr(key)
                snapshot_mapping[snapshot_key] = _capture_value_fallback(item, seen=seen)
            return snapshot_mapping

        if isinstance(value, list):
            return [_capture_value_fallback(item, seen=seen) for item in value]

        if isinstance(value, tuple):
            items = tuple(_capture_value_fallback(item, seen=seen) for item in value)
            if hasattr(value, "_fields"):
                try:
                    return type(value)(*items)
                except Exception:
                    return items
            return items

        if isinstance(value, set):
            return sorted((_capture_value_fallback(item, seen=seen) for item in value), key=repr)

        return repr(value)
    finally:
        seen.discard(value_id)


_PICKLE_SAFE = (bool, int, float, str, bytes, type(None))


def operator_config(op: Any) -> dict[str, Any]:
    """Return public instance attributes of *op* for tooltip/tracing display.

    Non-serializable values (callables, arbitrary objects) are converted to
    their repr() so the result is always safe to pickle.
    """
    try:
        attrs = vars(op)
    except TypeError:
        return {}
    return {
        k: v if isinstance(v, _PICKLE_SAFE) else repr(v)
        for k, v in attrs.items()
        if not k.startswith("_") and k != "pipeline"
    }


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


def _update_optional_mean(current: float | None, incoming: float | None, n: int) -> float | None:
    if incoming is None:
        return current
    if current is None or n <= 1:
        return incoming
    return current + (incoming - current) / n


def accumulate_trace_mean(avg: InvocationTrace, incoming: InvocationTrace, n: int) -> None:
    """Update *avg* in-place with an incremental mean over *incoming* traces."""
    avg.total_duration_s += (incoming.total_duration_s - avg.total_duration_s) / n
    avg.batch_size = _update_optional_mean(avg.batch_size, incoming.batch_size, n)
    if incoming.workers is not None:
        avg.workers = incoming.workers

    incoming_by_label = {s.label: s for s in incoming.spans}

    for span in avg.spans:
        inc = incoming_by_label.get(span.label)
        if inc is None:
            continue
        span.duration_s += (inc.duration_s - span.duration_s) / n
        span.error = span.error or inc.error
        if span.child_trace is not None and inc.child_trace is not None:
            span.child_trace.workers = inc.child_trace.workers
            accumulate_trace_mean(span.child_trace, inc.child_trace, n)
        elif span.child_trace is None and inc.child_trace is not None:
            child = InvocationTrace(
                batch_size=inc.child_trace.batch_size,
                workers=inc.child_trace.workers,
            )
            span.child_trace = child
            accumulate_trace_mean(child, inc.child_trace, 1)

    existing_labels = {s.label for s in avg.spans}
    for label, inc in incoming_by_label.items():
        if label in existing_labels:
            continue
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
            error=inc.error,
            operator_config=inc.operator_config,
            attributes=inc.attributes,
            input_shape=inc.input_shape,
            output_shape=inc.output_shape,
            output_value=inc.output_value,
            child_trace=child,
            operator_type=inc.operator_type,
        ))
        if child is not None:
            accumulate_trace_mean(child, inc.child_trace, 1)


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
            error=any(s.error for s in group),
            operator_config=group[0].operator_config,
            attributes=copy.deepcopy(group[0].attributes),
            input_shape=group[0].input_shape,
            output_shape=group[0].output_shape,
            output_value=group[0].output_value,
            child_trace=merge_traces([s.child_trace for s in group if s.child_trace is not None]) if any(s.child_trace for s in group) else None,
            operator_type=group[0].operator_type,
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
    def _raw_shape_of(obj: Any) -> Any | None:
        return getattr(obj, "shape", None)

    def _looks_like_torch_tensor(obj: Any) -> bool:
        module_name = type(obj).__module__
        return module_name.startswith("torch") and _raw_shape_of(obj) is not None

    def _shape_of(obj: Any) -> Any | None:
        shape = _raw_shape_of(obj)
        if shape is None:
            return None
        if _looks_like_torch_tensor(obj):
            return tuple(shape)
        return shape

    def _device_of(obj: Any) -> str | None:
        if not _looks_like_torch_tensor(obj):
            return None
        device = getattr(obj, "device", None)
        if device is None:
            return None
        return str(device)

    def _fmt_shape_with_device(shape: Any, device: str | None) -> str:
        if device is None:
            return str(shape)
        return f"{shape} @ {device}"

    name = type(value).__name__
    # ImagePayload, TensorPayload (have .array.shape)
    if hasattr(value, "array") and hasattr(value.array, "shape"):
        return f"{name} {_fmt_shape_with_device(_shape_of(value.array), _device_of(value.array))}"
    # bare torch.Tensor
    if _looks_like_torch_tensor(value):
        return f"{name} {_fmt_shape_with_device(_shape_of(value), _device_of(value))}"
    # bare numpy array
    if hasattr(value, "shape"):
        return f"{name} {_shape_of(value)}"
    # TensorRegistry: one "key: shape" entry per tensor
    if hasattr(value, "_tensors") and isinstance(getattr(value, "_tensors", None), dict):
        entries = ", ".join(
            f"{k}: {_fmt_shape_with_device(_shape_of(v), _device_of(v))}"
            for k, v in value._tensors.items()
        )
        return f"{name} {{{entries}}}"
    # RuntimeOutputs: named output tensors
    if hasattr(value, "names") and hasattr(value, "tensors"):
        entries = ", ".join(
            f"{n}: {_fmt_shape_with_device(_shape_of(t.array), _device_of(t.array))}"
            for n, t in zip(value.names, value.tensors)
        )
        return f"{name} {{{entries}}}"
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
