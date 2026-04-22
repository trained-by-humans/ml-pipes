# Tracing

ml-pipes includes a built-in tracing system that records per-operator latency
for every pipeline call. It is **opt-in and zero-overhead** when not configured
— pipelines without a collector run at exactly the same speed as before.

## Primitives

| Name | What it is |
|---|---|
| `StepSpan` | One operator step: label, start time, duration, error flag, optional input/output shapes |
| `InvocationTrace` | One complete pipeline call: ordered list of `StepSpan`s, total duration |
| `TraceCollector` | Interface with a single `on_trace(trace)` method — implement this to consume traces |
| `TracingConfig` | Groups collector + optional operator labels + optional shape capture |

## How it works

Each `pipeline(value)` call builds an `InvocationTrace` entirely on the call
stack — no locks, no shared state. Each operator step appends a `StepSpan` with
its wall-time duration. After the pipeline returns its result to the caller, it
hands the finished trace to the registered collector via `on_trace(trace)`.

This means:

- **No contention during execution.** The trace object is local to each call.
  Concurrent threads each build their own trace independently.
- **One collector callback per pipeline call**, not one per operator. Whatever
  locking or I/O the collector does, it happens after the result is delivered.
- **Errors are traced too.** If an operator raises, its `StepSpan` has
  `error=True` and the trace is still delivered to the collector via the
  `on_trace` callback.

## Quick start

```python
from ml_pipes import Pipeline, PrintCollector, TracingConfig
from ml_pipes import Resize, Normalize, Infer, NMS, ToDetections  # ...

pipeline = Pipeline(
    [Resize((640, 640)), Normalize(), Infer("model.onnx"), NMS(), ToDetections()],
    tracing=TracingConfig(collector=PrintCollector()),
)

pipeline("image.jpg")
```

Output:

```
  0:Resize                          0.28ms  (  0.5%)
  1:Normalize                       2.22ms  (  4.6%)
  2:Infer                          39.19ms  (81.4%)
  3:NMS                             0.21ms  (  0.4%)
  4:ToDetections                    0.02ms  (  0.0%)
  total                            48.12ms
```

## Attaching and detaching at runtime

`set_tracing()` lets you enable, reconfigure, or disable tracing on an existing
pipeline without rebuilding it. This is useful for profiling a specific window
of calls without instrumenting the whole program.

```python
pipeline = Pipeline([...])

# enable — just pass the collector
pipeline.set_tracing(PrintCollector())
pipeline("image.jpg")

# disable
pipeline.set_tracing(None)
pipeline("image.jpg")   # no overhead, no trace produced
```

With shape capture and custom operator labels:

```python
pipeline.set_tracing(
    PrintCollector(),
    capture_shapes=True,
    operator_labels=["resize", "normalize", "infer", "nms", "to_detections"],
)
```

## Built-in collectors

### `PrintCollector`

Prints each trace to stdout. Good for development and one-off debugging.

```python
from ml_pipes import PrintCollector
pipeline.set_tracing(PrintCollector())
```

### `AggregateCollector`

Accumulates stats across multiple invocations — average latency and percentage
of total per operator. Useful for benchmarking steady-state throughput.

```python
from ml_pipes import AggregateCollector

agg = AggregateCollector()
pipeline.set_tracing(agg)

for image in image_batch:
    pipeline(image)

pipeline.set_tracing(None)
agg.print_summary()
```

Output:

```
  Calls : 100
  Avg pipeline latency : 46.53ms

  Operator                              Avg ms  % of total
  ----------------------------------- --------  ----------
  0:Resize                                0.18ms       0.4%
  1:Normalize                             2.22ms       4.8%
  2:Infer                                37.71ms      81.0%
  3:NMS                                   0.21ms       0.5%
  4:ToDetections                          0.01ms       0.0%
```

`AggregateCollector` also exposes the data programmatically:

```python
agg.total_calls                  # int
agg.avg_pipeline_latency_ms      # float
agg.avg_operator_latency_ms()    # dict[label, float]
agg.operator_fractions()         # dict[label, float]  — fraction of avg pipeline latency
agg.reset()                      # clear all accumulated state
```

## Writing a custom collector

Subclass `TraceCollector` and implement `on_trace`. The collector receives a
complete `InvocationTrace` after every pipeline call.

```python
from ml_pipes import TraceCollector, InvocationTrace

class SlowCallAlert(TraceCollector):
    def __init__(self, threshold_ms: float) -> None:
        self._threshold_s = threshold_ms / 1000

    def on_trace(self, trace: InvocationTrace) -> None:
        slow = [s for s in trace.spans if s.duration_s > self._threshold_s]
        if slow:
            labels = ", ".join(s.label for s in slow)
            print(f"[SLOW] {trace.total_duration_s*1000:.1f}ms total — slow steps: {labels}")

pipeline.set_tracing(SlowCallAlert(threshold_ms=50.0))
```

## Batch pipelines

Batch regions produce a nested child `InvocationTrace` attached to the
`Batch` span. Every invocation — whether it was the batch leader or a follower
— receives the same `Batch` span with the same child trace, so every trace is
structurally complete and self-contained.

```
InvocationTrace:
  StepSpan("0:Resize",         0.2ms)
  StepSpan("1:Batch[wait]",   18.2ms)   ← this thread's wait for the batch to form
  StepSpan("1:Batch",         34.7ms)   ← leader's region, shared with all followers
    ↳ child InvocationTrace [batch_size=8]:
        StepSpan("2:Collate",    0.1ms)
        StepSpan("3:Infer",     34.5ms)
        StepSpan("4:Distribute", 0.1ms)
  StepSpan("5:NMS",            0.2ms)
```

`Batch[wait]` captures each thread's own gate wait time — the leader's wait is
short (just accumulation time), while a follower's wait includes the full region
execution. `batch_size` on the child trace lets an aggregator normalise region
latency per sample.

> [!TIP]
> The warm-up run is always slower — ONNX Runtime JIT-compiles the graph on
> first use and the OS cold-starts file I/O. Use `AggregateCollector` over
> several runs and discard the first call, or run a dedicated warm-up before
> starting the measurement window.

## OpenTelemetry export

An optional bridge to OpenTelemetry is available in
`ml_pipes.collectors.otel_collector`. It requires the `otel` extra:

```bash
pip install ml-pipes[otel]
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry import trace

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

from ml_pipes.collectors.otel_collector import OtelCollector
pipeline.set_tracing(OtelCollector())
pipeline("image.jpg")
```

Each pipeline call becomes a root OTel span with one child span per operator.
Batch regions produce nested child spans. The bridge is not imported by
`ml_pipes` itself — there is no implicit dependency on the OTel SDK.

## See also

- `examples/run_yolo8n_tracing.py` — full end-to-end tracing example with
  both `PrintCollector` and `AggregateCollector` on the YOLOv8n pipeline
- `PERFORMANCE.md` — throughput and batching guidance
