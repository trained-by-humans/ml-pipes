# ml-pipes

Build explicit ML pipelines you can validate, run, inspect, trace, and
benchmark.

`ml-pipes` is a framework for composing explicit ML pipelines. It consists of
a core pipeline runtime and a set of installable domain-specific operator
packages, plus shared pipeline tooling built around explicit operator
boundaries.

## Quick Start

`ml-pipes` requires Python 3.10+.

To run one example as fast as possible, install only the stack it needs and
execute it directly:

```bash
pip install 'ml-pipes[onnx,vision]'
python examples/run_yolo8_onnx.py
```

To pick the matching install for other runnable examples, see
[examples/README.md](examples/README.md). For the full package matrix and the
public component import model, see [docs/PACKAGES.md](docs/PACKAGES.md).

## Supported Use-cases/Domains

`ml-pipes` supports multiple domains through installable packages. For the
current package coverage, install profiles, and public imports, see
[docs/PACKAGES.md](docs/PACKAGES.md).

One concrete example is vision inference with `ml-pipes[vision,onnx]`, as
shown in [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py):

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Extract, Infer
from ml_pipes.standard import Pick, Recall, Store
from ml_pipes.tensor import ArgMax, GatherRows, Slice, Squeeze, Transpose
from ml_pipes.vision import (
    ConvertBoxFormat,
    Detections,
    ImagePayload,
    NMS,
    Normalize,
    ProjectBoxes,
    Resize,
    ToDetections,
)


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
            GatherRows("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=conf_threshold),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ],
        auto_validate=True,
    )
```

Most examples run with minimal setup out of the box. See
[examples/README.md](examples/README.md) for runnable entry points and any
example-specific setup.

## Why ml-pipes

- **Keep the whole flow visible.** Preprocessing, model calls, postprocessing,
  and downstream application logic stay in one explicit pipeline.
- **Use one tooling model everywhere.** Validation, inspection, tracing, and
  benchmarking all read the same operator boundaries.
- **Adapt different model families.** Wrap ONNX, Torch, or your own runtime
  without changing the surrounding scaffolding pattern.
- **Compose larger systems.** Merge or embed pipelines to build services,
  endpoints, data-preparation flows, and larger ML applications.

For the design rationale and internal structure behind this model, see
[docs/DESIGN.md](docs/DESIGN.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Key Features

Your part is to define operators and compose them into a pipeline. For
example, you can define one small operator and drop it into a pipeline that
already has a few steps:

```python
from ml_pipes.core import Operator, Pipeline


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

Try it: [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py)

### Inspection

Inspection runs the pipeline once and gives you a step-by-step view of what
each operator produced.

The same explicit operator boundaries drive the inspection report, starting
with a pipeline-wide overview and extending into step-level interactions.

```python
from ml_pipes.inspection import PipelineInspector

result = pipeline.inspect(sample)
result.dump("inspection.pkl")
PipelineInspector().show_in_browser(result, orientation="horizontal")
```

You can store the inspection result, or show it in a browser:

<p align="center">
  <img
    src=".github/assets/yolo8_tiled_inspection_overview.png"
    alt="Full static overview of a tiled inspection report"
    width="100%">
</p>

Inspection also supports larger outputs and drill-down interactions:

<table width="100%">
  <tr>
    <td width="59%" align="center" valign="top">
      <img
        src=".github/assets/yolo8_tiled_inspection_overview_scroll.gif"
        alt="Scrolling overview of a tiled inspection report">
      <br>
      <sub>Scrollable overview</sub>
    </td>
    <td width="41%" align="center" valign="top">
      <img
        src=".github/assets/yolo8_tiled_tiles_click.gif"
        alt="Before and after tile click in the inspection report">
      <br>
      <sub>Tile drill-down</sub>
    </td>
  </tr>
</table>


Try it: [examples/run_inspect.py](examples/run_inspect.py)

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

Try it: [examples/run_yolo8_tracing.py](examples/run_yolo8_tracing.py)

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

Try it: [examples/benchmarks/run_yolo8_benchmark.py](examples/benchmarks/run_yolo8_benchmark.py)

## Where To Go Next

- **Browse the documentation** in [docs/README.md](docs/README.md).
- **Start from runnable examples** in [examples/README.md](examples/README.md).
- **Run a baseline vision pipeline** with [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py).
- **Inspect and debug one run** with
  [examples/run_inspect.py](examples/run_inspect.py).
- **Build a service or larger app** with
  [examples/run_yolo8_endpoint.py](examples/run_yolo8_endpoint.py) and
  [docs/COMPOSITION.md](docs/COMPOSITION.md).
- **Wrap your own model** with
  [docs/SCAFFOLDING.md](docs/SCAFFOLDING.md).
