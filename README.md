# ml-pipes

Compute-composition framework for building ML systems around
explicit data flow.

## Install

`ml-pipes` requires Python 3.10+.

```bash
pip install -e .
pip install -e .[torch]  # optional Torch operators
pip install -e .[otel]   # optional OpenTelemetry collector support
pip install -e .[dev]    # tests and type checking
```

## Quick start

A pipeline for running YOLO inference with `ml-pipes` can look like this:

```python
from ml_pipes import (
    ArgMax, ConvertBoxFormat, Decode, GatherScores, Infer,
    NMS, Normalize, Pick, Pipeline, ProjectBoxes, Recall,
    Resize, Extract, Slice, Squeeze, Store, ToDetections, Transpose,
)

pipeline = Pipeline([
    Decode(),
    Resize((640, 640)),
    Store("resize_transform", source=1),
    Pick(0),
    Normalize(),
    Infer("yolov8n.onnx"),
    Extract("output0", as_="preds"),
    Squeeze("preds"),
    Transpose("preds"),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="scores"),
    ArgMax("scores", as_="classes"),
    GatherScores("scores", "classes"),
    ConvertBoxFormat(from_="cxcywh"),
    NMS(),
    Recall("resize_transform"),
    ProjectBoxes(),
    ToDetections(),
])

detections = pipeline("image.jpg")
print(detections.boxes, detections.scores, detections.classes)
```

You can also inspect the static pipeline shape without running it:

```python
repr(pipeline)
pipeline.describe(show_defaults=True)
```

## Documentation

- [docs/OPERATORS.md](docs/OPERATORS.md) — what operators are and how to write them
- [docs/COMPOSITION.md](docs/COMPOSITION.md) — building pipelines and composing pipelines together
- [docs/SCAFFOLDING.md](docs/SCAFFOLDING.md) — tutorial for wrapping a model in a composable pipeline scaffold
- [docs/DESIGN.md](docs/DESIGN.md) — design rationale and comparison with other approaches
- [docs/VALIDATION.md](docs/VALIDATION.md) — contract validation, strict mode, and boundary tightening
- [docs/TRACING.md](docs/TRACING.md) — traces, collectors, and the runtime observation model inspection builds on
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) — repeated-run measurement, sweeps, saved artifacts, and CLI benchmarking
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — tuning concurrency, batching, and serialization
- [docs/TORCH.md](docs/TORCH.md) — Torch operators, boundaries, and examples
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the internal ownership map of the framework

## Operator Packages

`ml-pipes` is designed to work with your own code. A pipeline can be built from packaged
operators, local operators, or ordinary callables as long as their boundaries
compose. When you do need a new operator, follow
[docs/OPERATORS.md](docs/OPERATORS.md).

In practice, start from the closest existing operator package or module and
reuse as much as possible before adding new code. That gives you clearer
validation, inspection, tracing, and less one-off logic to maintain.

| Package | Focus | References |
|---|---|---|
| `ml_pipes` / `ml_pipes.ops` | Built-in operators for image decoding, inference, tensor registries, regions, context, and postprocessing | [docs/OPERATORS.md](docs/OPERATORS.md) |
| `ml_pipes.data_ops` | Iterable and data-preparation operators for filtering, mapping, deduplication, and shaping | [run_sms_spam_prepare.py](examples/run_sms_spam_prepare.py) |
| `ml_pipes.torch` | Optional Torch operators for explicit NumPy-to-Torch handoffs, device movement, and Torch-native postprocessing | [docs/TORCH.md](docs/TORCH.md), [run_mask2former_torch_postprocess.py](examples/torch/run_mask2former_torch_postprocess.py), [run_mask2former_numpy_postprocess.py](examples/torch/run_mask2former_numpy_postprocess.py) |

All of these surfaces use the same `Pipeline`, validation, tracing,
inspection, and benchmarking APIs. The Torch examples also require
`transformers` and `safetensors` in the environment.

## Defining an Operator

An operator can be as simple as a plain callable for one-off logic or a class
with `__call__` for reusable configuration.

For example:

```python
from ml_pipes import Pipeline


class StripText:
    def __call__(self, text: str) -> str:
        return text.strip()


class SplitWords:
    def __init__(self, delimiter: str = " "):
        self.delimiter = delimiter

    def __call__(self, text: str) -> list[str]:
        return [part for part in text.split(self.delimiter) if part]


pipeline = Pipeline([StripText(), SplitWords()])
print(pipeline("  red blue green  "))
```

> [!TIP]
> Add accurate `__call__` annotations so validation can reason about the boundary.

For more advanced forms such as context operators, region operators, or custom
`resolve_contract(...)`, see [docs/OPERATORS.md](docs/OPERATORS.md). For validation
rules, see [docs/VALIDATION.md](docs/VALIDATION.md).

## How To Use Pipelines

### Wrap a Pipeline Behind an Interface

One common pattern is to keep a pipeline behind a small function or class that
owns the `Pipeline` instance. That works well for local model calls, service
objects, or API handlers:

```python
class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self._pipeline = Pipeline([
            Decode(),
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
        ])

    def __call__(self, image_path: str) -> Detections:
        return self._pipeline(image_path)

detector = YoloDetector("yolov8n.onnx", conf_threshold=0.3)
detections = detector("image.jpg")
```

### Compose an Application From Pipelines

Another common pattern is to build a larger workflow by composing smaller
pipelines. `a + b` merges pipelines into one flat operator list with shared
context. `a >> b` connects one pipeline to another while keeping each child
pipeline as its own boundary.

For example:

```python
preprocess = Pipeline([Decode(), Resize((640, 640)), Normalize()])
detect = Pipeline([Infer("detector.onnx")])
postprocess = Pipeline([Extract("output0", as_="preds"), ToDetections()])

vision = preprocess + detect + postprocess

embed = Pipeline([Decode(), Normalize(), Infer("embedder.onnx")])
classify = Pipeline([Infer("classifier.onnx"), ArgMax()])

service = embed >> classify
```

For composition semantics, API forms, and validation after composition, see
[docs/COMPOSITION.md](docs/COMPOSITION.md).

## Common Use Cases

### Model Scaffolding

If you need to wrap a model so it composes inside a larger pipeline or app,
start from a scaffold: an explicit sequence of steps around the model
boundary. The runtime boundary can be ONNX, Torch, or your own callable
around another library; the important part is to keep the rest of the flow
explicit: map raw outputs into semantic tensors, adapt layout, handle quirks,
normalize coordinates, filter and reduce candidates, and project back to the
source image. That makes the integration easier to validate, inspect, debug,
benchmark, adapt, and reuse across model variants.

For the full walkthrough, see
[docs/SCAFFOLDING.md](docs/SCAFFOLDING.md).

## Examples

See [examples/README.md](examples/README.md) for the full index.

Highlighted entry points:

- [examples/run_yolo8_onnx.py](examples/run_yolo8_onnx.py) — baseline YOLO ONNX detection pipeline
- [examples/run_rfdetr_nano.py](examples/run_rfdetr_nano.py) — DETR-style detector with normalized-box postprocessing
- [examples/run_inspect.py](examples/run_inspect.py) — step-by-step inspection of one pipeline run
- [examples/run_sms_spam_prepare.py](examples/run_sms_spam_prepare.py) — non-vision data preparation example
- [examples/benchmarks/](examples/benchmarks) and [examples/streaming/](examples/streaming) — benchmark and live-inference workflows
