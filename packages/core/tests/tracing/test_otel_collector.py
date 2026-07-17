from __future__ import annotations

from types import SimpleNamespace

from ml_pipes.collectors.otel_collector import OtelCollector
from ml_pipes.tracing import InvocationTrace, StepSpan


class _FakeSpan:
    def __init__(self, name: str, sink: list["_FakeSpan"]) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}
        self.status: object | None = None
        sink.append(self)

    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status = status


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_as_current_span(self, name: str) -> _FakeSpan:
        return _FakeSpan(name, self.spans)


def test_otel_collector_omits_shape_attributes() -> None:
    tracer = _FakeTracer()
    collector = object.__new__(OtelCollector)
    collector._tracer = tracer
    collector._otel_trace = SimpleNamespace(
        Status=lambda code: ("status", code),
        StatusCode=SimpleNamespace(ERROR="ERROR"),
    )

    collector._collect(
        InvocationTrace(
            spans=[
                StepSpan(
                    label="0:step",
                    start_time=0.0,
                    duration_s=0.012,
                    input_shape="ndarray (3, 4)",
                    output_shape="ndarray (3, 4)",
                )
            ],
            total_duration_s=0.012,
        )
    )

    root_span, step_span = tracer.spans

    assert root_span.name == "pipeline"
    assert root_span.attributes == {"total_duration_ms": 12.0}

    assert step_span.name == "0:step"
    assert step_span.attributes == {"duration_ms": 12.0}
