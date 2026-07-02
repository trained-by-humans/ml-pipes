from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

from ..tracing import InvocationTrace, StepSpan


class InspectionResult:
    """The result of Pipeline.inspect(): raw span data, no display logic."""

    def __init__(self, spans: list[StepSpan]) -> None:
        self.spans = spans

    def __repr__(self) -> str:
        lines = ["InspectionResult:"]
        self._repr_spans(self.spans, lines, indent=2)
        return "\n".join(lines)

    @staticmethod
    def _repr_spans(spans: list[StepSpan], lines: list[str], indent: int) -> None:
        prefix = " " * indent
        for span in spans:
            shape = span.output_shape or ""
            err = " [ERROR]" if span.error else ""
            lines.append(f"{prefix}{span.label:35s}  {str(shape):20s}{err}")
            if span.child_trace is not None:
                InspectionResult._repr_spans(span.child_trace.spans, lines, indent + 2)

    def _repr_html_(self) -> str:
        """Jupyter auto-render hook — uses default PipelineInspector."""
        from .inspector import PipelineInspector

        return PipelineInspector().to_html(self)

    def dump(self, path: str | Path) -> Path:
        """Serialize this result to a file."""
        return InspectionSerializer().dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "InspectionResult":
        """Load a serialized result from a file."""
        return InspectionSerializer().load(path)


class InspectionSerializer:
    """Serializes / deserializes an InspectionResult to bytes via pickle."""

    def dumps(self, result: InspectionResult) -> bytes:
        return pickle.dumps(self._sanitize(result))

    @staticmethod
    def _sanitize(result: InspectionResult) -> InspectionResult:
        """Return a copy with operator_type cleared so locally-defined classes don't break pickle."""
        return InspectionResult([InspectionSerializer._sanitize_span(span) for span in result.spans])

    @staticmethod
    def _sanitize_span(span: StepSpan) -> StepSpan:
        child = None
        if span.child_trace is not None:
            child = InvocationTrace(
                spans=[InspectionSerializer._sanitize_span(child_span) for child_span in span.child_trace.spans],
                total_duration_s=span.child_trace.total_duration_s,
                batch_size=span.child_trace.batch_size,
                workers=span.child_trace.workers,
            )
        return dataclasses.replace(span, operator_type=None, child_trace=child)

    def loads(self, data: bytes) -> InspectionResult:
        obj = pickle.loads(data)
        if not isinstance(obj, InspectionResult):
            raise TypeError(f"Expected InspectionResult, got {type(obj).__name__}")
        return obj

    def dump(self, result: InspectionResult, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.dumps(result))
        return out

    def load(self, path: str | Path) -> InspectionResult:
        return self.loads(Path(path).read_bytes())
