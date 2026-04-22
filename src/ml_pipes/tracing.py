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


@dataclass
class InvocationTrace:
    spans: list[StepSpan] = field(default_factory=list)
    total_duration_s: float = 0.0
    batch_size: int | None = None

    def span_fractions(self) -> dict[str, float]:
        if self.total_duration_s == 0.0:
            return {s.label: 0.0 for s in self.spans}
        return {s.label: s.duration_s / self.total_duration_s for s in self.spans}


@dataclass
class TracingConfig:
    collector: TraceCollector
    operator_labels: list[str] | None = None
    capture_shapes: bool = False


class TraceCollector(ABC):
    @abstractmethod
    def on_trace(self, trace: InvocationTrace) -> None: ...


def _extract_shape(value: Any) -> tuple | None:
    if hasattr(value, "array") and hasattr(value.array, "shape"):
        return tuple(value.array.shape)
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None
