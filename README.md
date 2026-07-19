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

## Pipeline Tooling

Your code can stay small. One step can be as simple as loading a file:

```python
from pathlib import Path

from ml_pipes.core import Operator

@Operator
class LoadFile:
    def __call__(self, image_path: str | Path) -> bytes:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()
```

Compose that step into a pipeline, then let the tooling below work on it. This
shortened example just shows where your code fits:

```python
from pathlib import Path

from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall, Store
from ml_pipes.vision import Decode, DrawBoxes, SaveImage

pipeline = Pipeline(
    [
        LoadFile(),
        Decode(),
        Store("source_image"),
        # ... YOLOv8 preprocessing, inference, and postprocessing ...
        Recall("source_image", prepend=True),
        DrawBoxes(),
        SaveImage(Path("annotated.jpg"), at=0),
    ]
)
```

The examples below use the full YOLOv8 pipeline from
[examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py).

### Validation

Validation checks the operator boundaries and tells you what the pipeline
accepts and returns.

```python
contract = pipeline.validate()
print(contract.input_type)
print(contract.output_type)
```

```text
str | pathlib.Path
tuple[ml_pipes.vision.types.ImagePayload, ml_pipes.vision.types.Detections]
```

Try it: [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py)

### Inspection

Inspection runs the pipeline once and shows what each operator produced.

The browser report starts with a pipeline-wide overview and lets you drill into
individual steps.

```python
from ml_pipes.inspection import PipelineInspector

image_path = "image.jpg"
result = pipeline.inspect(image_path)
result.dump("inspection.pkl")
PipelineInspector().show_in_browser(result, orientation="horizontal")
```

You can save the inspection result and open it in a browser:

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

Tracing records one call through the same pipeline and prints the per-step
runtime breakdown. You can also aggregate repeated calls or monitor a live
stream.

```python
from ml_pipes.collectors import AggregateCollector, PrintCollector

image_path = "image.jpg"
pipeline.set_tracing(PrintCollector())
pipeline(image_path)
pipeline.set_tracing(None)
```

```text
  0:LoadFile                        0.31ms  ( 1.9%)
  1:Decode                          2.18ms  (13.0%)
  2:Store                           0.01ms  ( 0.0%)
  3:Resize                          0.21ms  ( 1.3%)
  4:Store                           0.01ms  ( 0.1%)
  5:Pick                            0.02ms  ( 0.1%)
  6:Normalize                       2.22ms  (13.2%)
  7:Infer                           7.79ms  (46.5%)
  8:Extract                         0.06ms  ( 0.3%)
  9:Squeeze                         0.04ms  ( 0.2%)
  10:Transpose                      0.02ms  ( 0.1%)
  11:Slice                          0.02ms  ( 0.1%)
  12:Slice                          0.01ms  ( 0.1%)
  13:ArgMax                         1.11ms  ( 6.6%)
  14:GatherRows                     0.10ms  ( 0.6%)
  15:ConvertBoxFormat               0.09ms  ( 0.5%)
  16:NMS                            0.22ms  ( 1.3%)
  17:Recall                         0.00ms  ( 0.0%)
  18:ProjectBoxes                   0.05ms  ( 0.3%)
  19:ToDetections                   0.02ms  ( 0.1%)
  20:Recall                         0.00ms  ( 0.0%)
  21:DrawBoxes                      0.21ms  ( 1.3%)
  22:SaveImage                      1.86ms  (11.1%)
  total                            16.76ms
```

An `AggregateCollector` can roll up repeated calls into summary metrics:

```python
agg = AggregateCollector()
image_path = "image.jpg"
pipeline.set_tracing(agg)
for _ in range(4):
    pipeline(image_path)
pipeline.set_tracing(None)
agg.flush()

print(f"Calls: {agg.total_calls}")
print(f"Latency Avg.: {agg.avg_pipeline_latency_ms:.2f}ms")
```
```text
Calls: 4
Latency Avg.: 16.76ms

0:LoadFile                        0.31ms  ( 1.9%)
...
```

`ThroughputCollector(target_fps=30.0)` adds a rolling status line on top of the
same trace stream:

```text
FPS[1.0s]: 13.0 / 30.0 (43%) / FPS[20s]: 12.8 / 30.0 (43%) / latency: 57.9ms / CPU: 7% MEM: 1.6GB (5%)
```

Try it: [examples/run_yolo8_tracing.py](examples/run_yolo8_tracing.py)

### Benchmarking

Benchmarking repeats the same pipeline and produces a summary table you can
print, save, or diff against a baseline.

```python
from ml_pipes.benchmark import Benchmark, MeasurementConfig

image_path = "image.jpg"
config = MeasurementConfig(runs=30, warmup=5, percentiles=(0.50, 0.95, 0.99))

result = Benchmark(
    pipeline=pipeline,
    input_fn=lambda: ("image.jpg", image_path, None, None),
    measurement=config,
    label="yolo8n",
).run()

result.save("results/yolo8n.json")
print(result.to_table())
```

```text
operator                   mean        p50        p95        p99     stddev        min        max
-------------------------------------------------------------------------------------------------
total                    16.04      15.84      17.24      17.37       0.65      15.10      17.41
0:LoadFile                0.32       0.28       0.40       0.93       0.16       0.23       1.14
1:Decode                  2.12       2.10       2.23       2.28       0.06       2.03       2.29
...
7:Infer                   7.42       7.31       8.23       8.73       0.46       6.90       8.89
...
21:DrawBoxes              0.19       0.20       0.21       0.24       0.02       0.14       0.25
22:SaveImage              1.91       1.86       2.16       2.22       0.16       1.70       2.24
-------------------------------------------------------------------------------------------------
runs: 30  (all values in ms)
```

Compare saved results to spot regressions or measure the impact of switching
from `yolo8s` to `yolo8n`:

```python
from ml_pipes.benchmark import BenchmarkResult

baseline = BenchmarkResult.load("results/yolo8s.json")
candidate = BenchmarkResult.load("results/yolo8n.json")

print(baseline.diff(candidate).to_table())
```

```text
baseline : yolo8s
candidate: yolo8n
-----------------------------------------------------------------------------------------------------
operator                    Δmean        Δmean%          Δp50          Δp95          Δp99        note
-----------------------------------------------------------------------------------------------------
total                     -3.44ms       -18.49%       -3.17ms       -5.92ms       -6.33ms
0:LoadFile                -0.07ms       -18.66%       -0.07ms       -0.15ms       +0.10ms
...
7:Infer                   -2.51ms       -26.27%       -2.03ms       -4.85ms       -4.60ms
...
21:DrawBoxes              -0.07ms       -35.61%       -0.08ms       -0.05ms       -0.06ms
22:SaveImage              -0.16ms        -8.00%       -0.18ms       -0.35ms       -0.55ms
-----------------------------------------------------------------------------------------------------
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
