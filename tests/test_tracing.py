from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from ml_pipes import (
    Batch,
    InspectionSerializer,
    InvocationTrace,
    LazyPerItem,
    Pipeline,
    PrintCollector,
    SHORT_CIRCUIT,
    StepSpan,
    StreamItems,
    TraceCollector,
    TracingConfig,
    UnBatch,
)
from ml_pipes.tracing import PendingSpan, freeze_trace
from ml_pipes.types import TensorPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Capture(TraceCollector):
    def __init__(self) -> None:
        self.traces: list[InvocationTrace] = []

    def on_trace(self, trace: InvocationTrace) -> None:
        self.traces.append(trace)


class _ClosableSource:
    def __init__(self, values: list[int], *, fail_on_close: bool = False) -> None:
        self._values = iter(values)
        self.fail_on_close = fail_on_close
        self.close_calls = 0

    def __iter__(self) -> "_ClosableSource":
        return self

    def __next__(self) -> int:
        return next(self._values)

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_on_close:
            raise RuntimeError("close failed")


def _double(x: int) -> int:
    return x * 2


def _add_one(x: int) -> int:
    return x + 1


def _short_circuit(x: int) -> object:
    del x
    return SHORT_CIRCUIT


def _failing(x: int) -> int:
    raise ValueError("boom")


def _make_pipeline(ops: list[Any], **kw) -> tuple[Pipeline[Any, Any], _Capture]:
    cap = _Capture()
    p = Pipeline(ops, tracing=TracingConfig(collector=cap, **kw))
    return p, cap


# ---------------------------------------------------------------------------
# Collector behaviour
# ---------------------------------------------------------------------------

def test_collector_called_once_per_invocation():
    p, cap = _make_pipeline([_double, _add_one])
    p(1)
    p(2)
    assert len(cap.traces) == 2


def test_lazy_top_level_trace_delivered_immediately_as_unmaterialized() -> None:
    p, cap = _make_pipeline([LazyPerItem(), _double, StreamItems()])

    result = p([1, 2, 3])

    assert len(cap.traces) == 1
    assert cap.traces[0].spans[0].duration_s == 0.0
    assert cap.traces[0].spans[0].attributes == {}
    assert cap.traces[0].spans[0].child_trace is None
    assert list(result) == [2, 4, 6]
    assert cap.traces[0].spans[0].attributes == {}
    assert cap.traces[0].spans[0].child_trace is None


def test_freeze_trace_closes_pending_spans_and_returns_detached_step_spans() -> None:
    child_pending = PendingSpan(
        label="[0]",
        start_time=0.0,
        duration_s=0.01,
        attributes={"seen": 1},
    )
    parent_pending = PendingSpan(
        label="0:LazyPerItem",
        start_time=0.0,
        duration_s=0.02,
        attributes={"seen": 1, "emitted": 1},
        child_trace=InvocationTrace(
            spans=[child_pending],
            total_duration_s=0.01,
        ),
    )
    trace = InvocationTrace(spans=[parent_pending], total_duration_s=0.02)

    frozen = freeze_trace(trace)

    assert isinstance(frozen.spans[0], StepSpan)
    assert frozen.spans[0].attributes == {"seen": 1, "emitted": 1}
    assert frozen.spans[0].child_trace is not None
    assert isinstance(frozen.spans[0].child_trace.spans[0], StepSpan)
    assert frozen.spans[0].child_trace.spans[0].attributes == {"seen": 1}
    assert parent_pending.is_closed is True
    assert child_pending.is_closed is True

    parent_pending.attributes = {"seen": 9}
    child_pending.duration_s = 9.0

    assert frozen.spans[0].attributes == {"seen": 1, "emitted": 1}
    assert frozen.spans[0].child_trace.spans[0].duration_s == 0.01


def test_lazy_top_level_close_only_closes_source_once() -> None:
    source = _ClosableSource([1, 2, 3])
    p, cap = _make_pipeline([LazyPerItem(), _double, StreamItems()])

    result = p(source)
    assert len(cap.traces) == 1
    assert cap.traces[0].spans[0].attributes == {}
    assert next(result) == 2

    result.close()

    assert source.close_calls == 1
    assert len(cap.traces) == 1
    assert cap.traces[0].spans[0].attributes == {}
    assert cap.traces[0].spans[0].child_trace is None


def test_lazy_top_level_close_failure_does_not_rewrite_delivered_trace() -> None:
    source = _ClosableSource([1, 2, 3], fail_on_close=True)
    p, cap = _make_pipeline([LazyPerItem(), _double, StreamItems()])

    result = p(source)
    assert len(cap.traces) == 1
    assert cap.traces[0].spans[0].attributes == {}
    assert next(result) == 2

    result.close()

    assert source.close_calls == 1
    assert len(cap.traces) == 1
    assert cap.traces[0].spans[0].error is False
    assert cap.traces[0].spans[0].attributes == {}
    assert cap.traces[0].spans[0].child_trace is None


def test_spans_ordered_and_labelled():
    p, cap = _make_pipeline([_double, _add_one])
    p(1)
    assert [s.label for s in cap.traces[0].spans] == ["0:_double", "1:_add_one"]


def test_custom_operator_labels():
    p, cap = _make_pipeline([_double, _add_one],
                             operator_labels=["double", "add_one"])
    p(1)
    assert [s.label for s in cap.traces[0].spans] == ["double", "add_one"]


def test_error_span_flagged():
    p, cap = _make_pipeline([_double, _failing])
    with pytest.raises(ValueError, match="boom"):
        p(1)
    spans = cap.traces[0].spans
    assert not spans[0].error
    assert spans[1].error


def test_collector_error_does_not_crash_pipeline():
    class _BrokenCollector(TraceCollector):
        def on_trace(self, trace: InvocationTrace) -> None:
            raise RuntimeError("collector is broken")

    p = Pipeline([_double], tracing=TracingConfig(collector=_BrokenCollector()))
    assert p(3) == 6


def test_error_trace_delivered_to_collector():
    p, cap = _make_pipeline([_double, _failing])
    with pytest.raises(ValueError):
        p(1)
    assert len(cap.traces) == 1


def test_error_trace_contains_completed_and_error_spans():
    p, cap = _make_pipeline([_double, _failing, _add_one])
    with pytest.raises(ValueError):
        p(1)
    trace = cap.traces[0]
    # _double completed successfully before the error
    assert any(s.label.endswith("_double") and not s.error for s in trace.spans)
    # _failing is flagged as an error span
    assert any(s.label.endswith("_failing") and s.error for s in trace.spans)
    # _add_one never ran — should not appear
    assert not any(s.label.endswith("_add_one") for s in trace.spans)


def test_inspect_short_circuit_stops_downstream_and_round_trips() -> None:
    pipeline = Pipeline([_double, _short_circuit, _add_one])

    result = pipeline.inspect(3)

    assert [span.label for span in result.spans] == ["0:_double", "1:_short_circuit"]
    assert result.spans[1].output_value is SHORT_CIRCUIT

    restored = InspectionSerializer().loads(InspectionSerializer().dumps(result))

    assert [span.label for span in restored.spans] == ["0:_double", "1:_short_circuit"]
    assert restored.spans[1].output_value is SHORT_CIRCUIT


def test_set_tracing_window():
    p = Pipeline([_double])
    cap = _Capture()
    p(1)
    assert cap.traces == []

    p.set_tracing(cap)
    p(1)
    assert len(cap.traces) == 1

    p.set_tracing(None)
    p(1)
    assert len(cap.traces) == 1


def test_set_tracing_mid_call_does_not_crash():
    import time

    def _slow(x: int) -> int:
        time.sleep(0.02)
        return x

    p, cap = _make_pipeline([_double, _slow, _add_one])

    def disable_mid_call():
        time.sleep(0.01)
        p.set_tracing(None)

    t = threading.Thread(target=disable_mid_call)
    t.start()
    result = p(3)
    t.join()

    assert result == 7
    assert len(cap.traces) == 1


def test_span_fractions_bounded():
    p, cap = _make_pipeline([_double, _add_one])
    p(3)
    fracs = cap.traces[0].span_fractions()
    assert all(0.0 <= v <= 1.0 for v in fracs.values())


# ---------------------------------------------------------------------------
# Shape capture
# ---------------------------------------------------------------------------

def test_shapes_off_by_default():
    p, cap = _make_pipeline([_double])
    p(5)
    assert cap.traces[0].spans[0].input_shape is None
    assert cap.traces[0].spans[0].output_shape is None


def test_shapes_recorded_for_ndarray():
    arr = np.zeros((3, 4))

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _make_pipeline([_passthrough], capture_shapes=True)
    result = p(arr)
    assert result is arr
    span = cap.traces[0].spans[0]
    assert span.input_shape == "ndarray (3, 4)"
    assert span.output_shape == "ndarray (3, 4)"


def test_shapes_recorded_for_tensor_payload():
    payload = TensorPayload(array=np.zeros((1, 3, 640, 640)), layout="NCHW", dtype="float32")

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _make_pipeline([_passthrough], capture_shapes=True)
    result = p(payload)
    assert result is payload
    assert cap.traces[0].spans[0].input_shape == "TensorPayload (1, 3, 640, 640)"


# ---------------------------------------------------------------------------
# Batch — traced path span structure
# ---------------------------------------------------------------------------

def _make_batch_pipeline(capture: _Capture) -> Pipeline[int, int]:
    def _identity_batch(x: list[int]) -> list[int]:
        return x

    return Pipeline(
        [Batch(size=2, timeout=1.0), _identity_batch, UnBatch(), _add_one],
        tracing=TracingConfig(collector=capture),
    )


def _run_two_threads(pipeline: Pipeline[int, int]) -> list[int]:
    results = [None, None]

    def run(idx, val):
        results[idx] = pipeline(val)

    t1 = threading.Thread(target=run, args=(0, 1))
    t2 = threading.Thread(target=run, args=(1, 2))
    t1.start(); t2.start()
    t1.join(); t2.join()
    return [int(result) for result in results]


def test_batch_wait_span_present_on_all_threads():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    for trace in cap.traces:
        assert any("[wait]" in s.label for s in trace.spans)


def test_batch_region_span_has_child_trace():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    all_batch_spans = [s for t in cap.traces for s in t.spans if s.child_trace is not None]
    assert len(all_batch_spans) >= 1
    assert isinstance(all_batch_spans[0].child_trace, InvocationTrace)


def test_batch_child_trace_has_batch_size():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    for trace in cap.traces:
        for span in trace.spans:
            if span.child_trace is not None:
                assert span.child_trace.batch_size == 2


def test_batch_child_trace_contains_operator_spans():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    for trace in cap.traces:
        for span in trace.spans:
            if span.child_trace is not None:
                assert len(span.child_trace.spans) > 0


def test_batch_leader_and_follower_span_labels_identical():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    assert len(cap.traces) == 2
    label_sets = [tuple(s.label for s in t.spans) for t in cap.traces]
    assert label_sets[0] == label_sets[1]


def test_batch_follower_wait_excludes_batch_duration():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    # The follower's raw gate.enter() duration exceeds the batch region duration
    # (it blocks for lobby + batch). After subtracting batch duration, the
    # corrected wait should be shorter than the batch span on at least one trace.
    corrected_waits = [
        next(s.duration_s for s in t.spans if "[wait]" in s.label)
        for t in cap.traces
    ]
    batch_durations = [
        next(s.duration_s for s in t.spans if s.child_trace is not None)
        for t in cap.traces
    ]
    assert min(corrected_waits) < max(batch_durations)


def test_batch_follower_wait_span_present_on_leader_error():
    def _failing_batch(x: list[Any]) -> list[Any]:
        raise ValueError("batch boom")

    cap = _Capture()
    p = Pipeline(
        [Batch(size=2, timeout=1.0), _failing_batch, UnBatch(), _add_one],
        tracing=TracingConfig(collector=cap),
    )
    errors = [None, None]

    def run(idx, val):
        try:
            p(val)
        except Exception as e:
            errors[idx] = e

    t1 = threading.Thread(target=run, args=(0, 1))
    t2 = threading.Thread(target=run, args=(1, 2))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert all(isinstance(e, ValueError) for e in errors)
    # Every trace must have both a wait span and a batch region span.
    for trace in cap.traces:
        assert any("[wait]" in s.label for s in trace.spans)
        batch_spans = [s for s in trace.spans if "[wait]" not in s.label and "Batch" in s.label]
        assert len(batch_spans) == 1
        assert batch_spans[0].error
        # The failing child span is recorded inside the child trace.
        assert batch_spans[0].child_trace is not None
        assert any(s.error for s in batch_spans[0].child_trace.spans)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_calls_each_get_own_trace():
    p, cap = _make_pipeline([_double, _add_one])
    n_threads = 10
    barrier = threading.Barrier(n_threads)

    def run():
        barrier.wait()
        p(1)

    threads = [threading.Thread(target=run) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cap.traces) == n_threads
    assert len({id(t) for t in cap.traces}) == n_threads


# ---------------------------------------------------------------------------
# merge_traces — captured field preservation
# ---------------------------------------------------------------------------

def test_merge_traces_preserves_captured_fields():
    from ml_pipes.tracing import StepSpan, InvocationTrace, merge_traces

    span_a = StepSpan(
        label="0:op", start_time=0.0, duration_s=0.01,
        operator_config={"size": 640},
        input_shape="ndarray (3, 4)",
        output_shape="ndarray (3, 4)",
        output_value=42,
    )
    span_b = StepSpan(
        label="0:op", start_time=0.0, duration_s=0.03,
        operator_config={"size": 640},
        input_shape="ndarray (3, 4)",
        output_shape="ndarray (3, 4)",
        output_value=42,
    )
    trace_a = InvocationTrace(spans=[span_a], total_duration_s=0.01)
    trace_b = InvocationTrace(spans=[span_b], total_duration_s=0.03)

    merged = merge_traces([trace_a, trace_b])

    assert len(merged.spans) == 1
    s = merged.spans[0]
    assert s.operator_config == {"size": 640}
    assert s.input_shape == "ndarray (3, 4)"
    assert s.output_shape == "ndarray (3, 4)"
    assert s.output_value == 42
    assert s.duration_s == pytest.approx(0.02)


def test_merge_traces_propagates_error_flag():
    from ml_pipes.tracing import StepSpan, InvocationTrace, merge_traces

    ok   = StepSpan(label="0:op", start_time=0.0, duration_s=0.01, error=False)
    fail = StepSpan(label="0:op", start_time=0.0, duration_s=0.01, error=True)
    merged = merge_traces([
        InvocationTrace(spans=[ok],   total_duration_s=0.01),
        InvocationTrace(spans=[fail], total_duration_s=0.01),
    ])
    assert merged.spans[0].error is True


def test_merge_traces_preserves_attributes_without_item_metric_synthesis():
    from ml_pipes.tracing import StepSpan, InvocationTrace, merge_traces

    trace_a = InvocationTrace(
        spans=[StepSpan(label="0:op", start_time=0.0, duration_s=0.01, attributes={"kind": "first"})],
        total_duration_s=0.01,
    )
    trace_b = InvocationTrace(
        spans=[StepSpan(label="0:op", start_time=0.0, duration_s=0.03, attributes={"kind": "second"})],
        total_duration_s=0.03,
    )

    merged = merge_traces([trace_a, trace_b])

    assert merged.spans[0].attributes == {"kind": "first"}


# ---------------------------------------------------------------------------
# PrintCollector smoke test
# ---------------------------------------------------------------------------

def test_print_collector_does_not_raise(capsys):
    p = Pipeline([_double, _add_one], tracing=TracingConfig(collector=PrintCollector()))
    result = p(3)
    assert result == 7
    out = capsys.readouterr().out
    assert "0:_double" in out
    assert "1:_add_one" in out
    assert "total" in out


# ---------------------------------------------------------------------------
# operator_config — pickle safety
# ---------------------------------------------------------------------------

def test_operator_config_callable_serializable():
    import pickle
    from ml_pipes.tracing import operator_config

    class OpWithLambda:
        def __init__(self):
            self.threshold = 0.5
            self.predicate = lambda x: x > 0  # not pickle-safe as raw value

    cfg = operator_config(OpWithLambda())
    assert cfg["threshold"] == 0.5
    assert isinstance(cfg["predicate"], str)   # converted to repr
    # must not raise
    pickle.dumps(cfg)


def test_merge_traces_preserves_captured_fields():
    from ml_pipes.tracing import StepSpan, InvocationTrace, merge_traces

    def make_trace(val: Any) -> InvocationTrace:
        span = StepSpan(
            label="1:op",
            start_time=0.0,
            duration_s=0.01,
            error=False,
            operator_config={"k": "v"},
            input_shape="(1,)",
            output_shape="(2,)",
            output_value=val,
        )
        t = InvocationTrace()
        t.spans.append(span)
        return t

    merged = merge_traces([make_trace("a"), make_trace("b")])
    s = merged.spans[0]
    assert s.output_value == "a"       # first worker's value is representative
    assert s.output_shape == "(2,)"
    assert s.input_shape == "(1,)"
    assert s.operator_config == {"k": "v"}
    assert not s.error


def test_merge_traces_propagates_error_flag():
    from ml_pipes.tracing import StepSpan, InvocationTrace, merge_traces

    def make_trace(err: bool) -> InvocationTrace:
        t = InvocationTrace()
        t.spans.append(StepSpan(label="1:op", start_time=0.0, duration_s=0.01, error=err))
        return t

    merged = merge_traces([make_trace(False), make_trace(True)])
    assert merged.spans[0].error is True


def test_inspect_with_lambda_operator_is_serializable():
    import pickle
    from ml_pipes import FilterPredictions, InspectionSerializer
    from ml_pipes.types import Detections

    pred = Detections(boxes=[[0,0,1,1],[1,1,2,2]], scores=[0.9, 0.4], classes=[0, 1])
    p = Pipeline([FilterPredictions(predicate=lambda d: [s > 0.5 for s in d.scores])])
    result = p.inspect(pred)

    data = InspectionSerializer().dumps(result)
    restored = InspectionSerializer().loads(data)
    assert len(restored.spans) == len(result.spans)
