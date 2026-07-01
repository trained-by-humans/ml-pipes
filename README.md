# ml-pipes

Build explicit ML pipelines you can run, inspect, validate, trace, and
benchmark.

`ml-pipes` is a compute-composition framework for ML systems built around
explicit data flow. It keeps preprocessing, runtime calls, postprocessing, and
application logic visible in one pipeline instead of spreading them across
wrappers and hidden control flow.

The same operator boundaries power validation, inspection, tracing, and
benchmarking, so the pipeline you write is also the pipeline your tools
understand.

## Quick Start

`ml-pipes` requires Python 3.10+. Run commands from the repository root.

```bash
pip install -e .
# Optional extras
pip install -e .[torch]  # Torch operators
pip install -e .[otel]   # OpenTelemetry collector support
pip install -e .[dev]    # tests and type checking
```

Run a baseline detection pipeline:

```bash
python examples/run_yolo8_onnx.py
```

The core inference pipeline in
[examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py) looks like this:

```python
def yolo8_inference_pipeline(
    model_path: Path,
    conf_threshold: float = 0.25,
) -> Pipeline[ImagePayload, Detections]:
    return Pipeline(
        [
            Resize((640, 640)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(model_path),
            Extract("output0", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=conf_threshold),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ],
        auto_validate=True,
    )
```

Most self-contained examples download the models and sample assets they need
into the shared `examples/.example_assets/` cache on demand. Generic entry points
such as `examples/run_detection.py` expect explicit inputs instead. For more
runnable entry points, including the equivalent `examples/`-local commands,
see [examples/README.md](examples/README.md).

## Why ml-pipes

- **Keep the whole flow visible.** Preprocessing, model calls, postprocessing,
  and downstream application logic stay in one explicit pipeline.
- **Use one tooling model everywhere.** Validation, inspection, tracing, and
  benchmarking all read the same operator boundaries.
- **Adapt different model families.** Wrap ONNX, Torch, or your own runtime
  without changing the surrounding scaffolding pattern.
- **Compose larger systems.** Merge or embed pipelines to build services,
  endpoints, data-preparation flows, and larger ML applications.

## Key Features

Your part is to define operators and compose them into a pipeline. For
example, you can define one small operator and drop it into a pipeline that
already has a few steps:

```python
from ml_pipes.core import (
    Operator,
    Pipeline,
)


def strip_text(text: str) -> str:
    return text.strip()


def split_words(text: str) -> list[str]:
    return text.split()


@Operator
class Lowercase:
    def __call__(self, text: str) -> str:
        return text.lower()


pipeline = Pipeline([strip_text, Lowercase(), split_words])
sample = "  Hello World  "
```

### Validation

Validation checks the operator boundaries in the pipeline and returns the
contract the whole pipeline exposes.

```python
contract = pipeline.validate()
print(contract)
```

```text
TypeContract(input_type=<class 'str'>, output_type=list[str])
```

### Inspection

Inspection runs the pipeline once and gives you a step-by-step view of what
each operator produced.

```python
result = pipeline.inspect(sample)
print(result)
```

```text
InspectionResult:
  0:strip_text                         str
  1:Lowercase                          str
  2:split_words                        list [2]
```

### Tracing

Tracing records one call through the same pipeline and shows the per-step
runtime breakdown.

```python
from ml_pipes.collectors import PrintCollector

pipeline.set_tracing(PrintCollector())
pipeline(sample)
pipeline.set_tracing(None)
```

```text
  0:strip_text                      0.01ms  (23.4%)
  1:Lowercase                       0.02ms  (42.6%)
  2:split_words                     0.01ms  (17.1%)
  total                             0.04ms
```

### Benchmarking

Benchmarking repeats that same pipeline over many runs and returns a summary
table you can print, diff, or save.

```python
from ml_pipes.benchmark import Benchmark, MeasurementConfig

result = Benchmark(
    pipeline=pipeline,
    input_fn=lambda: ("sample", sample, None, None),
    measurement=MeasurementConfig(runs=5, warmup=1, percentiles=(0.50,)),
    label="text-pipeline",
).run()
print(result.to_table())
```

```text
operator             mean        p50     stddev        min        max
---------------------------------------------------------------------
total               0.03       0.03       0.00       0.03       0.03
0:strip_text        0.01       0.01       0.00       0.01       0.01
1:Lowercase         0.01       0.01       0.00       0.01       0.02
2:split_words       0.01       0.01       0.00       0.01       0.01
```

## Start From A Concrete Goal

- **Run a baseline vision pipeline** with
  [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py).
- **Inspect and debug one run** with
  [examples/run_inspect.py](examples/run_inspect.py).
- **Wrap your own model** with
  [docs/SCAFFOLDING.md](docs/SCAFFOLDING.md).
- **Build a service or larger app** with
  [examples/run_yolo8_endpoint.py](examples/run_yolo8_endpoint.py) and
  [docs/COMPOSITION.md](docs/COMPOSITION.md).
- **See a non-vision pipeline** with
  [examples/run_sms_spam_prepare.py](examples/run_sms_spam_prepare.py).

## Where To Go Next

- [examples/README.md](examples/README.md) — full runnable example index
- [docs/SCAFFOLDING.md](docs/SCAFFOLDING.md) — wrap a new model in a pipeline
- [docs/OPERATORS.md](docs/OPERATORS.md) — reuse or define operators
- [docs/COMPOSITION.md](docs/COMPOSITION.md) — compose pipelines into larger
  applications
- [docs/DESIGN.md](docs/DESIGN.md) and
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — understand the rationale and
  internal structure
- [docs/README.md](docs/README.md) — full documentation index
