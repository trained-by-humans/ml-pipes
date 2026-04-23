from __future__ import annotations

import threading

from ml_pipes import (
    ConcurrentCollector,
    InvocationTrace,
    Pipeline,
    SerialCollector,
    TracingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SerialCapture(SerialCollector):
    def __init__(self) -> None:
        super().__init__()
        self.traces: list[InvocationTrace] = []

    def _collect(self, trace: InvocationTrace) -> None:
        self.traces.append(trace)


class _ConcurrentCapture(ConcurrentCollector):
    def __init__(self) -> None:
        super().__init__()
        self.traces: list[InvocationTrace] = []

    def _collect(self, trace: InvocationTrace) -> None:
        self.traces.append(trace)


def _double(x: int) -> int:
    return x * 2


# ---------------------------------------------------------------------------
# SerialCollector
# ---------------------------------------------------------------------------

def test_serial_result_correct():
    cap = _SerialCapture()
    p = Pipeline([_double], tracing=TracingConfig(collector=cap))
    assert p(3) == 6
    assert len(cap.traces) == 1


def test_serial_concurrent_calls_no_lost_increments():
    cap = _SerialCapture()
    p = Pipeline([_double], tracing=TracingConfig(collector=cap))
    n = 20
    barrier = threading.Barrier(n)

    def run():
        barrier.wait()
        p(1)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(cap.traces) == n


# ---------------------------------------------------------------------------
# ConcurrentCollector
# ---------------------------------------------------------------------------

def test_concurrent_result_not_blocked():
    import time

    class _SlowCapture(_ConcurrentCapture):
        def _collect(self, trace: InvocationTrace) -> None:
            time.sleep(0.05)
            super()._collect(trace)

    with _SlowCapture() as cap:
        p = Pipeline([_double], tracing=TracingConfig(collector=cap))
        t0 = time.perf_counter()
        result = p(3)
        elapsed = time.perf_counter() - t0

    assert result == 6
    assert elapsed < 0.05
    assert len(cap.traces) == 1


def test_concurrent_traces_delivered_after_flush():
    cap = _ConcurrentCapture()
    p = Pipeline([_double], tracing=TracingConfig(collector=cap))
    p(1)
    p(2)
    cap.flush()
    cap.stop()
    assert len(cap.traces) == 2


def test_concurrent_context_manager_flushes_on_exit():
    with _ConcurrentCapture() as cap:
        p = Pipeline([_double], tracing=TracingConfig(collector=cap))
        for i in range(5):
            p(i)
    assert len(cap.traces) == 5


def test_concurrent_stop_idempotent():
    with _ConcurrentCapture() as cap:
        Pipeline([_double], tracing=TracingConfig(collector=cap))(1)
    cap.stop()
