from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from ml_pipes import (
    Batch,
    InvocationTrace,
    Pipeline,
    PrintCollector,
    StepSpan,
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


def _capture_pipeline(*ops, **kw) -> tuple[Pipeline, _Capture]:
    cap = _Capture()
    cfg = TracingConfig(collector=cap, **kw)
    p = Pipeline(list(ops), tracing=cfg)
    return p, cap


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_no_tracing_zero_overhead():
    p = Pipeline([_double, _add_one])
    assert p(3) == 7
    assert p._tracing_config is None


def test_collector_called_once_per_invocation():
    p, cap = _capture_pipeline(_double, _add_one)
    p(1)
    p(2)
    assert len(cap.traces) == 2


def test_spans_ordered_and_labelled():
    p, cap = _capture_pipeline(_double, _add_one)
    p(1)
    labels = [s.label for s in cap.traces[0].spans]
    assert labels == ["0:_double", "1:_add_one"]


def test_total_duration_positive():
    p, cap = _capture_pipeline(_double)
    p(5)
    assert cap.traces[0].total_duration_s > 0


def test_custom_operator_labels():
    p, cap = _capture_pipeline(_double, _add_one, operator_labels=["double", "add_one"])
    p(1)
    labels = [s.label for s in cap.traces[0].spans]
    assert labels == ["double", "add_one"]


def test_error_span_flagged_and_exception_propagates():
    p, cap = _capture_pipeline(_double, _failing)
    with pytest.raises(ValueError, match="boom"):
        p(1)
    spans = cap.traces[0].spans
    assert spans[0].label == "0:_double" and not spans[0].error
    assert spans[1].label == "1:_failing" and spans[1].error


def test_set_tracing_after_construction():
    p = Pipeline([_double])
    cap = _Capture()
    p(1)  # no tracing yet
    assert cap.traces == []

    p.set_tracing(cap)
    p(1)
    assert len(cap.traces) == 1

    p.set_tracing(None)
    p(1)
    assert len(cap.traces) == 1  # unchanged


def test_span_fractions_sum():
    p, cap = _capture_pipeline(_double, _add_one)
    p(3)
    fracs = cap.traces[0].span_fractions()
    assert all(0.0 <= v <= 1.0 for v in fracs.values())


# ---------------------------------------------------------------------------
# Shape capture
# ---------------------------------------------------------------------------

def test_shapes_off_by_default():
    p, cap = _capture_pipeline(_double)
    p(5)
    assert cap.traces[0].spans[0].input_shape is None
    assert cap.traces[0].spans[0].output_shape is None


def test_shapes_recorded_for_ndarray():
    arr = np.zeros((3, 4))

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _capture_pipeline(_passthrough, capture_shapes=True)
    p(arr)
    span = cap.traces[0].spans[0]
    assert span.input_shape == (3, 4)
    assert span.output_shape == (3, 4)


def test_shapes_recorded_for_tensor_payload():
    payload = TensorPayload(array=np.zeros((1, 3, 640, 640)), layout="NCHW", dtype="float32")

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _capture_pipeline(_passthrough, capture_shapes=True)
    p(payload)
    span = cap.traces[0].spans[0]
    assert span.input_shape == (1, 3, 640, 640)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _identity_batch(x: list[Any]) -> list[Any]:
    return x


def _make_batch_pipeline(capture: _Capture) -> Pipeline:
    cfg = TracingConfig(collector=capture)
    return Pipeline(
        [Batch(size=2, timeout=1.0), _identity_batch, UnBatch(), _add_one],
        tracing=cfg,
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
        wait_labels = [s.label for s in trace.spans if "[wait]" in s.label]
        assert len(wait_labels) == 1


def test_batch_region_span_has_child_trace():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    leader_traces = [t for t in cap.traces if any(s.child_trace is not None for s in t.spans)]
    assert len(leader_traces) >= 1
    batch_span = next(s for s in leader_traces[0].spans if s.child_trace is not None)
    assert isinstance(batch_span.child_trace, InvocationTrace)


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


def test_batch_follower_gets_leader_batch_span():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    assert len(cap.traces) == 2
    # Both traces should have a Batch span (with child_trace)
    for trace in cap.traces:
        batch_spans = [s for s in trace.spans if s.child_trace is not None]
        assert len(batch_spans) == 1


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
    wait_durations = []
    for trace in cap.traces:
        for span in trace.spans:
            if "[wait]" in span.label:
                wait_durations.append(span.duration_s)
    assert len(wait_durations) == 2
    # follower blocks until leader finishes region — its wait must be >= leader's
    assert max(wait_durations) > min(wait_durations)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_calls_each_get_own_trace():
    p, cap = _capture_pipeline(_double, _add_one)
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
    # Each trace must be a distinct object
    ids = {id(t) for t in cap.traces}
    assert len(ids) == n_threads


# ---------------------------------------------------------------------------
# PrintCollector smoke test
# ---------------------------------------------------------------------------

def test_print_collector_does_not_raise(capsys):
    cfg = TracingConfig(collector=PrintCollector())
    p = Pipeline([_double, _add_one], tracing=cfg)
    p(3)
    out = capsys.readouterr().out
    assert "0:_double" in out
    assert "1:_add_one" in out
    assert "total" in out
