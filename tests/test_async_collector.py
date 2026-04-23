from __future__ import annotations

import threading

from ml_pipes import AsyncCollector, InvocationTrace, Pipeline, TraceCollector, TracingConfig


class _Capture(TraceCollector):
    def __init__(self) -> None:
        self.traces: list[InvocationTrace] = []
        self._lock = threading.Lock()

    def on_trace(self, trace: InvocationTrace) -> None:
        with self._lock:
            self.traces.append(trace)


def _double(x: int) -> int:
    return x * 2


def test_result_not_blocked():
    """pipeline() returns before the collector processes the trace."""
    import time

    delivered = threading.Event()

    class _SlowCapture(TraceCollector):
        def on_trace(self, trace: InvocationTrace) -> None:
            time.sleep(0.05)
            delivered.set()

    with AsyncCollector(_SlowCapture()) as collector:
        p = Pipeline([_double], tracing=TracingConfig(collector=collector))
        t0 = time.perf_counter()
        result = p(3)
        elapsed = time.perf_counter() - t0

    assert result == 6
    assert elapsed < 0.05
    assert delivered.is_set()


def test_traces_delivered_after_flush():
    cap = _Capture()
    collector = AsyncCollector(cap)
    p = Pipeline([_double], tracing=TracingConfig(collector=collector))
    p(1)
    p(2)
    collector.flush()
    collector.stop()
    assert len(cap.traces) == 2


def test_context_manager_flushes_on_exit():
    cap = _Capture()
    with AsyncCollector(cap) as collector:
        p = Pipeline([_double], tracing=TracingConfig(collector=collector))
        for i in range(5):
            p(i)
    assert len(cap.traces) == 5


def test_concurrent_calls_all_delivered():
    cap = _Capture()
    with AsyncCollector(cap) as collector:
        p = Pipeline([_double], tracing=TracingConfig(collector=collector))
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


def test_stop_is_idempotent_after_context_manager():
    cap = _Capture()
    with AsyncCollector(cap) as collector:
        Pipeline([_double], tracing=TracingConfig(collector=collector))(1)
    collector.stop()  # second stop should not hang or raise
