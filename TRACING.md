# Tracing

ml-pipes includes a built-in tracing system that records per-operator latency
for every pipeline call. It is **opt-in** — pipelines without a collector
skip all collector calls, I/O, and synchronization.

## Primitives

| Name | What it is |
|---|---|
| `StepSpan` | One operator step: label, start time, duration, error flag, optional input/output shapes |
| `InvocationTrace` | One complete pipeline call: ordered list of `StepSpan`s, total duration |
| `TraceCollector` | Interface with a single `on_trace(trace)` method — implement this to consume traces |
| `TracingConfig` | Groups collector + optional operator labels + optional shape capture |

## How it works

Each `pipeline(value)` call builds an `InvocationTrace` entirely on the
calling thread — each operator step appends a `StepSpan` with its wall-time
duration to that thread's own trace object. After the pipeline returns its
result to the caller, it hands the finished trace to the collector via
`on_trace(trace)`.

Because tracing is confined to the calling thread:

- **No locking needed in the pipeline.** The trace is never shared between
  threads during execution, so there is nothing to protect. If the collector
  writes to shared state (a file, a metrics sink, a list), that is the only
  place a lock is needed — and it runs after the result is already returned.
- **One complete trace per invocation.** The collector receives a fully-formed
  `InvocationTrace` covering the entire call, so every invocation can be
  attributed, correlated, and measured in isolation.
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

> [!CAUTION]
> The first run is always slower and will skew measurements — ONNX Runtime
> JIT-compiles the graph on first use and the OS cold-starts file I/O. Run a
> dedicated warm-up call before starting the measurement window.

> [!TIP]
> Use `AggregateCollector` over several runs to get stable average latency
> per operator rather than relying on a single invocation.

## Configuration

`TracingConfig` groups all tracing options in one place:

```python
from ml_pipes import TracingConfig, PrintCollector

TracingConfig(
    collector=PrintCollector(),       # required — any TraceCollector implementation
    operator_labels=["resize", "infer", "nms"],  # override default "{i}:ClassName" labels
    capture_shapes=True,              # record input/output shapes on each StepSpan
)
```

All options are also available directly on `set_tracing()`:

```python
pipeline.set_tracing(
    PrintCollector(),
    operator_labels=["resize", "infer", "nms"],
    capture_shapes=True,
)
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


## Tracing Output

Each span shows the operator label, wall-time duration, and share of total. Error spans are marked with `!`:

```
  0:Resize                            0.28ms  (  0.5%)
  1:Normalize                         2.22ms  (  4.6%)
  2:Infer                            39.19ms  ( 81.4%) !
  3:NMS                               0.21ms  (  0.4%)
  total                              48.12ms
```

Batch regions produce a nested child trace:

```
  0:Resize                            0.20ms  (  0.5%)
  1:Batch[wait]                       0.50ms  (  1.3%)
  1:Batch                            34.70ms  ( 90.1%)
    ↳ child trace [batch_size=8]:
        2:_collate                    0.10ms  (  0.3%)
        3:Infer                      34.50ms  ( 99.4%)
        4:_distribute                 0.10ms  (  0.3%)
        total                        34.70ms
  5:NMS                               2.00ms  (  5.2%)
  total                              38.50ms
```

`Batch[wait]` captures each thread's lobby accumulation time — how long it waited
for enough samples to form a batch. Both leader and follower record only this
window; the batch region execution time is accounted for separately in the `Batch`
span. `batch_size` on the child trace lets an aggregator normalize region latency
per sample.

## Built-in collectors

### `PrintCollector`

Prints each trace to stdout. Good for development and one-off debugging.

```python
from ml_pipes import PrintCollector
pipeline.set_tracing(PrintCollector())
```

Output:

```
  0:Resize                            0.28ms  (  0.5%)
  1:Normalize                         2.22ms  (  4.6%)
  2:Infer                            39.19ms  ( 81.4%)
  3:NMS                               0.21ms  (  0.4%)
  4:ToDetections                      0.02ms  (  0.0%)
  total                              48.12ms
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

> New to OpenTelemetry? See the [Getting started guide](https://opentelemetry.io/docs/getting-started/).

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

## See also

- `examples/run_yolo8_tracing.py` — full end-to-end tracing example with
  both `PrintCollector` and `AggregateCollector` on the YOLOv8 pipeline
- `PERFORMANCE.md` — throughput and batching guidance
