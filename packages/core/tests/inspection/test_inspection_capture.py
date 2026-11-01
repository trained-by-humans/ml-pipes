from __future__ import annotations

from ml_pipes.core import Pipeline, SHORT_CIRCUIT
from ml_pipes.inspection import InspectionSerializer
from ml_pipes.tracing import InvocationTrace, TraceCollector


def _double(value: int) -> int:
    return value * 2


def _short_circuit(value: int) -> object:
    return SHORT_CIRCUIT


def _must_not_run(value: int) -> int:
    raise AssertionError("Inspection must stop after SHORT_CIRCUIT")


def test_inspect_short_circuit_stops_downstream_and_round_trips() -> None:
    pipeline = Pipeline([_double, _short_circuit, _must_not_run])

    result = pipeline.inspect(3)

    assert [span.label for span in result.spans] == ["Input", "0:_double", "1:_short_circuit"]
    assert result.spans[2].output_value is SHORT_CIRCUIT

    restored = InspectionSerializer().loads(InspectionSerializer().dumps(result))

    assert [span.label for span in restored.spans] == ["Input", "0:_double", "1:_short_circuit"]
    assert restored.spans[2].output_value is SHORT_CIRCUIT


def test_inspect_ignores_active_trace_collector() -> None:
    class _BrokenCollector(TraceCollector):
        def on_trace(self, trace: InvocationTrace) -> None:
            raise RuntimeError("collector is broken")

    pipeline = Pipeline([_double])
    pipeline.set_tracing(_BrokenCollector())

    result = pipeline.inspect(3)

    assert [span.label for span in result.spans] == ["Input", "0:_double"]
    assert result.spans[1].output_value == 6


def test_inspection_round_trip_preserves_callable_config_and_captured_output():
    class Apply:
        def __init__(self):
            self.transform = lambda value: value * 2

        def __call__(self, value):
            return self.transform(value)

    result = Pipeline([Apply()]).inspect(3)
    serializer = InspectionSerializer()

    restored = serializer.loads(serializer.dumps(result))

    assert [span.label for span in restored.spans] == ["Input", "0:Apply"]
    assert restored.spans[0].output_value == 3
    assert restored.spans[1].output_value == 6
    assert isinstance(restored.spans[1].operator_config["transform"], str)
    assert restored.spans[1].operator_config == result.spans[1].operator_config
    assert restored.spans[1].operator_type is None
    assert result.spans[1].operator_type is Apply
