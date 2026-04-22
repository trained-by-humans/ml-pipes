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


def _make_pipeline(ops: list, traced: bool, **kw) -> tuple[Pipeline, _Capture | None]:
    """Build a pipeline with or without a collector attached."""
    if traced:
        cap = _Capture()
        p = Pipeline(ops, tracing=TracingConfig(collector=cap, **kw))
        return p, cap
    return Pipeline(ops), None


# Fixture that parametrizes all correctness tests across both execution paths.
@pytest.fixture(params=[True, False], ids=["traced", "untraced"])
def traced(request) -> bool:
    return request.param


# ---------------------------------------------------------------------------
# Correctness — both paths must produce identical results
# ---------------------------------------------------------------------------

def test_result_correct(traced):
    p, _ = _make_pipeline([_double, _add_one], traced)
    assert p(3) == 7


def test_result_correct_after_set_tracing():
    p = Pipeline([_double, _add_one])
    assert p(3) == 7
    p.set_tracing(PrintCollector())
    assert p(3) == 7
    p.set_tracing(None)
    assert p(3) == 7


def test_error_propagates(traced):
    p, _ = _make_pipeline([_double, _failing], traced)
    with pytest.raises(ValueError, match="boom"):
        p(1)


def test_context_op_result_correct(traced):
    from ml_pipes import Store, Recall
    p, _ = _make_pipeline([Store("x"), Recall("x")], traced)
    assert p(42) == (42, 42)


def test_batch_result_correct(traced):
    def _identity_batch(x: list[Any]) -> list[Any]:
        return x

    ops = [Batch(size=2, timeout=1.0), _identity_batch, UnBatch(), _add_one]
    p, _ = _make_pipeline(ops, traced)
    results = [None, None]

    def run(idx, val):
        results[idx] = p(val)

    t1 = threading.Thread(target=run, args=(0, 1))
    t2 = threading.Thread(target=run, args=(1, 2))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert set(results) == {2, 3}


# ---------------------------------------------------------------------------
# Traced-path only — span structure and collector behaviour
# ---------------------------------------------------------------------------

def test_no_tracing_config_by_default():
    p = Pipeline([_double, _add_one])
    assert p._tracing_config is None


def test_collector_called_once_per_invocation():
    p, cap = _make_pipeline([_double, _add_one], traced=True)
    p(1)
    p(2)
    assert len(cap.traces) == 2


def test_spans_ordered_and_labelled():
    p, cap = _make_pipeline([_double, _add_one], traced=True)
    p(1)
    assert [s.label for s in cap.traces[0].spans] == ["0:_double", "1:_add_one"]


def test_total_duration_positive():
    p, cap = _make_pipeline([_double], traced=True)
    p(5)
    assert cap.traces[0].total_duration_s > 0


def test_custom_operator_labels():
    p, cap = _make_pipeline([_double, _add_one], traced=True,
                             operator_labels=["double", "add_one"])
    p(1)
    assert [s.label for s in cap.traces[0].spans] == ["double", "add_one"]


def test_error_span_flagged():
    p, cap = _make_pipeline([_double, _failing], traced=True)
    with pytest.raises(ValueError, match="boom"):
        p(1)
    spans = cap.traces[0].spans
    assert not spans[0].error
    assert spans[1].error


def test_error_trace_delivered_to_collector():
    p, cap = _make_pipeline([_double, _failing], traced=True)
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


def test_span_fractions_bounded():
    p, cap = _make_pipeline([_double, _add_one], traced=True)
    p(3)
    fracs = cap.traces[0].span_fractions()
    assert all(0.0 <= v <= 1.0 for v in fracs.values())


# ---------------------------------------------------------------------------
# Shape capture
# ---------------------------------------------------------------------------

def test_shapes_off_by_default():
    p, cap = _make_pipeline([_double], traced=True)
    p(5)
    assert cap.traces[0].spans[0].input_shape is None
    assert cap.traces[0].spans[0].output_shape is None


def test_shapes_recorded_for_ndarray():
    arr = np.zeros((3, 4))

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _make_pipeline([_passthrough], traced=True, capture_shapes=True)
    result = p(arr)
    assert result is arr
    span = cap.traces[0].spans[0]
    assert span.input_shape == (3, 4)
    assert span.output_shape == (3, 4)


def test_shapes_recorded_for_tensor_payload():
    payload = TensorPayload(array=np.zeros((1, 3, 640, 640)), layout="NCHW", dtype="float32")

    def _passthrough(x: Any) -> Any:
        return x

    p, cap = _make_pipeline([_passthrough], traced=True, capture_shapes=True)
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


def test_batch_follower_gets_leader_batch_span():
    cap = _Capture()
    p = _make_batch_pipeline(cap)
    _run_two_threads(p)
    assert len(cap.traces) == 2
    for trace in cap.traces:
        assert any(s.child_trace is not None for s in trace.spans)


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


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_calls_each_get_own_trace():
    p, cap = _make_pipeline([_double, _add_one], traced=True)
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
