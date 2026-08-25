# Tracing

## What Traces Are

A trace is the recorded execution of one pipeline call.

It captures the operator-by-operator view of that call, including:

- which operators ran
- how long each step took
- where a failure happened
- how time was distributed across a region such as `Batch` or `Scatter`

## Quick Example

```python
from ml_pipes.collectors import PrintCollector
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Infer
from ml_pipes.vision import Decode, NMS, Normalize, Resize


pipeline = Pipeline([
    Decode(),
    Resize((640, 640)),
    Normalize(),
    Infer("model.onnx"),
    NMS(),
])
pipeline.set_tracing(PrintCollector())

pipeline("image.jpg")

pipeline.set_tracing(None)
```

Output:

```text
  0:Decode                       2.10ms  (  4.3%)
  1:Resize                       0.28ms  (  0.6%)
  2:Normalize                    2.22ms  (  4.6%)
  3:Infer                       43.90ms  ( 90.0%)
  4:NMS                          0.21ms  (  0.4%)
  total                         48.73ms
```

## How Tracing Works

Tracing is opt-in. If no collector is attached, the pipeline runs without
producing traces.

When tracing is enabled:

1. each `pipeline(value)` call creates one `InvocationTrace`
2. each operator appends a `StepSpan`
3. region operators can attach a child trace for the enclosed execution
4. if an operator raises, the error span is still recorded
5. the completed trace is delivered to the collector

That means a collector always receives one whole-call trace, not partial step
events.

You enable or reconfigure tracing on an existing pipeline with
`set_tracing(...)`.

```python
from ml_pipes.collectors import PrintCollector


pipeline.set_tracing(PrintCollector())
```

The currently attached collector is exposed as `pipeline.trace_collector`.
That property is read-only; attach or remove collectors through
`set_tracing(...)`.

Tracing stays focused on runtime step timing, failures, and region structure.
If you need per-step outputs and richer debug snapshots, use
`Pipeline.inspect()` instead.

> [!NOTE]
> The trace delivered to the collector is a frozen snapshot, so collectors can
> store it safely without later pipeline mutations changing the recorded data.
> If a collector raises, the pipeline call still completes; the tracing error
> is logged and that trace is dropped.

## When Traces Are Useful

Tracing is useful when you want to:

- understand an individual call, such as seeing which operator dominates one
  slow call or confirming where a failure happened
- monitor a running system through collectors, such as watching live latency or
  throughput summaries or exporting execution data
- support performance tuning by seeing how time is distributed within calls

> [!TIP]
> Use [`Benchmark`](BENCHMARKING.md) when you want to measure repeated-run
> behavior or compare configuration changes.

## Built-in Collectors

Use the table below to choose a collector based on how traces are delivered and
what each collector exposes.

| Collector | Delivery | Best for | Provides |
|---|---|---|---|
| `PrintCollector` | sync | local debugging | `stdout`, `last_trace`, `print_trace()` |
| `CaptureCollector` | sync | one-shot programmatic inspection | `last_trace` |
| `AggregateCollector` | async | rolling summaries | `avg_trace`, `total_calls`, `avg_pipeline_latency_ms`, `print_summary()`, `reset()` |
| `ThroughputCollector` | async | live service monitoring | `fps`, `window_fps`, live status line, `print_summary()`, optional resource stats |
| `OtelCollector` | async | external observability | exported spans |

## Creating Custom Collectors

All collectors receive an `InvocationTrace`.

- `trace.total_duration_s` is the whole-call duration
- `trace.spans` is the ordered list of `StepSpan`
- a region span may contain `span.child_trace`
- the trace is already frozen before delivery

### Choose A Base

- `TraceCollector` is the lowest-level interface. Implement `on_trace(trace)`
  directly when you want full control.
- `SerialCollector` is the best default for in-memory state. It serializes
  access with a lock and calls your `_collect(trace)` method.
- `ConcurrentCollector` moves trace processing onto a background thread. Use it
  when collection may block, write files, send telemetry, or do heavier work.
  Call `flush()` before reading results and `stop()` when done, or use it as a
  context manager.

### Useful Starting Points

- `CaptureCollector` is a good base when you want `last_trace`.
- `AggregateCollector` is a good base when you want rolling averages or
  summaries.

### Minimal Example

```python
from ml_pipes.collectors import SerialCollector
from ml_pipes.tracing import InvocationTrace


class SlowTraceCollector(SerialCollector):
    def __init__(self, threshold_ms: float = 50.0) -> None:
        super().__init__()
        self.threshold_s = threshold_ms / 1000.0
        self.slow_traces: list[InvocationTrace] = []

    def _collect(self, trace: InvocationTrace) -> None:
        if trace.total_duration_s >= self.threshold_s:
            self.slow_traces.append(trace)
```

If the collector is going to do file I/O, telemetry export, or any other work
you do not want on the call path, switch the base to `ConcurrentCollector`
instead.
