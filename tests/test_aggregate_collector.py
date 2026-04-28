from __future__ import annotations

import threading

from ml_pipes import AggregateCollector, Pipeline, TracingConfig


def _double(x: int) -> int:
    return x * 2


def test_concurrent_on_trace_no_lost_increments():
    with AggregateCollector() as agg:
        p = Pipeline([_double], tracing=TracingConfig(collector=agg))
        n = 50
        barrier = threading.Barrier(n)

        def run():
            barrier.wait()
            p(1)

        threads = [threading.Thread(target=run) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert agg.total_calls == n
    assert len(agg.avg_trace.spans) == 1
