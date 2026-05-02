from __future__ import annotations

import threading

from ml_pipes import AggregateCollector, InvocationTrace, Pipeline, StepSpan, TracingConfig


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


def test_aggregate_collector_averages_child_trace_batch_size():
    with AggregateCollector() as agg:
        agg.on_trace(
            InvocationTrace(
                spans=[
                    StepSpan(
                        label="Batch",
                        start_time=0.0,
                        duration_s=0.1,
                        child_trace=InvocationTrace(batch_size=1, total_duration_s=0.1),
                    )
                ],
                total_duration_s=0.1,
            )
        )
        agg.on_trace(
            InvocationTrace(
                spans=[
                    StepSpan(
                        label="Batch",
                        start_time=0.0,
                        duration_s=0.2,
                        child_trace=InvocationTrace(batch_size=4, total_duration_s=0.2),
                    )
                ],
                total_duration_s=0.2,
            )
        )
        agg.flush()

    assert agg.avg_trace.spans[0].child_trace is not None
    assert agg.avg_trace.spans[0].child_trace.batch_size == 2.5
