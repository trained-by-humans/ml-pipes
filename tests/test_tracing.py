from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from ml_pipes import (
    Batch,
    InvocationTrace,
    Pipeline,
    PrintCollector,
    TraceCollector,
    TracingConfig,
    UnBatch,
)
from ml_pipes.types import TensorPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Capture(TraceCollector):
    def __init__(self) -> None:
        self.traces: list[InvocationTrace] = []

    def on_trace(self, trace: InvocationTrace) -> None:
        self.traces.append(trace)


def _double(x: int) -> int:
    return x * 2


def _add_one(x: int) -> int:
    return x + 1


def _failing(x: int) -> int:
    raise ValueError("boom")


def _make_pipeline(ops: list, **kw) -> tuple[Pipeline, _Capture]:
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
    assert span.input_shape == (3, 4)
    assert span.output_shape == (3, 4)


def test_shapes_recorded_for_tensor_payload():
    payload = TensorPayload(array=np.zeros((1, 3, 640, 640)), layout="NCHW", dtype="float32")

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _make_pipeline([_passthrough], capture_shapes=True)
    result = p(payload)
    assert result is payload
    assert cap.traces[0].spans[0].input_shape == (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# Batch — traced path span structure
# ---------------------------------------------------------------------------

def _make_batch_pipeline(capture: _Capture) -> Pipeline:
    def _identity_batch(x: list[Any]) -> list[Any]:
        return x

    return Pipeline(
        [Batch(size=2, timeout=1.0), _identity_batch, UnBatch(), _add_one],
        tracing=TracingConfig(collector=capture),
    )


def _run_two_threads(pipeline: Pipeline) -> list[Any]:
    results = [None, None]

    def run(idx, val):
        results[idx] = pipeline(val)

    t1 = threading.Thread(target=run, args=(0, 1))
    t2 = threading.Thread(target=run, args=(1, 2))
    t1.start(); t2.start()
    t1.join(); t2.join()
    return results


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


def test_batch_follower_wait_longer_than_leader():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    wait_durations = [
        s.duration_s
        for t in cap.traces
        for s in t.spans
        if "[wait]" in s.label
    ]
    assert len(wait_durations) == 2
    assert max(wait_durations) > min(wait_durations)


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
    # Every trace must have a wait span — including followers whose gate.enter() raised.
    for trace in cap.traces:
        wait_spans = [s for s in trace.spans if "[wait]" in s.label]
        assert len(wait_spans) == 1
    # The follower's wait span is flagged as an error (leader's is not).
    error_wait_spans = [
        s for t in cap.traces for s in t.spans if "[wait]" in s.label and s.error
    ]
    assert len(error_wait_spans) == 1


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
