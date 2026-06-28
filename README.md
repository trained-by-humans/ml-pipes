# ml-pipes

Composable ML pipelines built from explicit operator boundaries.

`ml-pipes` is a compute-composition framework built around data flow. A
pipeline is a sequence of operator boundaries that push data forward: one step
receives a value, transforms it, and hands the result to the next step. The
value might be an image, a tensor registry, a batch, a record, or any other
payload, but the main thing the framework cares about is how that data moves
and changes.

Once data becomes the first-class concern, the tooling naturally follows it.
Validation checks that operator boundaries connect, Inspection gives you a
built-in lineage view of what each step produced in one run, Tracing records
how a call moved through the pipeline, and Benchmark measures the same flow
across repeated runs.

That model fits ML systems better than an app design centered on objects or
services. In many ordinary applications, most of the code is side effects with
a little data mutation around them. In ML applications, the ratio is usually
reversed: most of the work is data mutation, with side effects mostly at the
edges when you load inputs, call runtimes, or save outputs.

Even models are not first-class in that sense. A model call is just one
operator in a larger data path; weights and biases matter because they affect
how data is transformed, not because the framework treats the model itself as
the center of the system.

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

## Design Principles

Many inference SDKs are built around inheritance. That feels natural at
first, but in a mature codebase the prediction path often ends up spread across
several layers of base classes:

```python
class BaseInference:
    def predict(self, image): ...

class Model(BaseInference):
    def preprocess(self, image): ...

class RuntimeModel(Model):
    def infer(self, tensor): ...

class OnnxModel(RuntimeModel):
    def session(self): ...

class DetectionModel(OnnxModel):
    def postprocess(self, outputs): ...

class YoloDetector(DetectionModel):
    def postprocess(self, outputs): ...

class DetrDetector(DetectionModel):
    def postprocess(self, outputs): ...
```

Nothing is wrong with any one layer in isolation. The problem is that to
understand one prediction you now have to jump through multiple base classes to
find preprocessing, runtime invocation, output decoding, and task-specific
postprocessing.

**ml-pipes takes the opposite approach: composition.**

A pipeline is a plain list of small, single-purpose operators. Each operator
does one thing and knows nothing about the model, task, or runtime. Different
models produce different outputs — but at some level of abstraction they all
produce boxes, scores, class indices, and optionally masks. The right operators
applied in the right order produce the right result regardless of model family.
Most of the pipeline stays the same; only the model-specific decoding changes.

```python
# YOLOv8n
Pipeline([
    Decode(), Resize((640, 640)), Store("resize_transform", index=1), Pick(0), Normalize(),
    Infer("yolov8n.onnx"),
    Extract("output0", as_="preds"),          
    Squeeze("preds"), Transpose("preds"),     
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="scores"),
    ArgMax("scores", as_="classes"),          
    GatherScores("scores", "classes"),        
    ConvertBoxFormat(from_="cxcywh"), NMS(), Recall("resize_transform"), ProjectBoxes(), ToDetections(),
])

# DETR-style detector — only the section between Infer and NMS changes
Pipeline([
    Decode(), ...                                               # same preprocessing pipeline as above
    Infer("detr_nano.onnx"),                                    # different postprocessing
    Extract("pred_boxes", "logits", as_=("boxes", "logits")),   #
    Squeeze("boxes"), Squeeze("logits"),                        # ← model-specific
    Softmax("logits"),                                          #
    ArgMax("logits", as_="classes"),                            #
    GatherScores("logits", "classes", as_="scores"),            #
    Scale("boxes", by=(640, 640, 640, 640)),                    #
    ConvertBoxFormat(from_="cxcywh"), ...                       # same postprocessing pipeline as above
                           
])
```

Switching from YOLOv8 to a DETR-style detector changes a few lines
(`Scale` for normalized boxes and different softmax/argmax handling); the rest
stays explicit, reusable, and testable.

### Why Function-Style Coding Fits ML Workflows

Once you treat ML as transformation over data and artifacts, function-style
code becomes the natural fit. Loading maps files to records. Cleaning maps
records to cleaner records. Feature builders map records to tensors. Batchers
map examples to batches. Models map tensors to tensors. Evaluators map
predictions and labels to metrics. Export steps map internal results to files,
tables, or API responses.

A neural network is just one compact example:
```python
import torch.nn as nn

# A network is a function: input tensor → output tensor.
# nn.Sequential makes the transformation sequence the primary artifact.
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 112 * 112, 10),
)
```

The same idea applies to the larger workflow around the model. An ML pipeline
maps one value or artifact set into another: raw files to validated records,
records to features, features to predictions, predictions to metrics, or
results to downstream actions. Even when a step has side effects, the useful
abstraction is still an explicit boundary with clear inputs, outputs, and
effects.

A pipeline makes the data transformation the primary artifact. Reading the
pipeline top to bottom tells you exactly what happens to the data, in order,
without indirection. There is no hidden state between steps: each operator
receives a value, returns a value, and has no memory of previous calls unless
its effect is made explicit. That matches how you reason about ML workflows:
ingest, validate, enrich, batch, score, evaluate, persist. The reasoning stays
directly visible in the code rather than being distributed across a class
hierarchy.

> [!IMPORTANT]
> This also means the pipeline is inspectable and debuggable at every boundary.
> Inserting a `print` function or a logging step at any position in the list
> shows you the exact value flowing through at that point. There are no private
> fields to dig into, no method override chain to follow.

For the operator model and composition semantics behind this style, see
[docs/OPERATORS.md](docs/OPERATORS.md) and [docs/COMPOSITION.md](docs/COMPOSITION.md).

## Comparison with other approaches

Below is the same task — YOLOv8n object detection on a single image — implemented
with three different approaches.

### Ultralytics

Ultralytics ships a high-level API tightly coupled to the YOLO model family:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict("image.jpg", conf=0.25, iou=0.45)

for r in results:
    print(r.boxes.xyxy)   # boxes in original image space
    print(r.boxes.cls)    # class indices
    print(r.boxes.conf)   # confidence scores
```

Three lines from model load to result. The entire preprocessing, inference, and
postprocessing pipeline runs inside `predict`. This is the right choice if you
are building exclusively with YOLO models and the default postprocessing meets
your needs.

The cost is opacity and lock-in. You cannot swap a preprocessing step, insert a
custom tensor operation, or reuse any of the internal logic with a non-YOLO model.
The pipeline is not a list of steps you control — it is a method you call.

### Raw ONNX Runtime

Without a framework, you own everything:

```python
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("yolov8n.onnx")

# Preprocess
img = cv2.imread("image.jpg")
orig_h, orig_w = img.shape[:2]
resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
tensor = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

# Infer
preds = session.run(None, {session.get_inputs()[0].name: tensor})[0]

# Adapt layout: (1, 84, 8400) → (8400, 84)
preds = preds.squeeze().T
boxes_cxcywh = preds[:, :4]
class_scores  = preds[:, 4:]
classes = class_scores.argmax(axis=1)
scores  = class_scores[np.arange(len(classes)), classes]

# Confidence filter
keep = scores >= 0.25
boxes_cxcywh, scores, classes = boxes_cxcywh[keep], scores[keep], classes[keep]

# cxcywh → xyxy
x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

# NMS
indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), 0.25, 0.45)
boxes_xyxy, scores, classes = boxes_xyxy[indices], scores[indices], classes[indices]

# Scale back to original image space
boxes_xyxy[:, [0, 2]] *= orig_w / 640
boxes_xyxy[:, [1, 3]] *= orig_h / 640
```

Maximum control, but every part is hand-written. Switching to a DETR-style
model means rewriting the entire preprocessing and postprocessing block from
scratch: confidence filtering, box conversion, NMS, and projection all get
repeated. Nothing here is reusable across model families.

### ml-pipes

```python
from ml_pipes import (
    ArgMax, ConvertBoxFormat, Decode, GatherScores, Infer,
    NMS, Normalize, Pick, Pipeline, ProjectBoxes, Recall,
    Resize, Extract, Slice, Squeeze, Store, ToDetections, Transpose,
)

pipeline = Pipeline([
    Decode(),
    Resize((640, 640)),
    Store("resize_transform", index=1),
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
```

More explicit than Ultralytics and more structured than raw ONNX. Every step is
named and individually testable. Switching to a DETR-style model changes three
operators (`Scale` for normalised boxes, `Softmax` before `ArgMax`); all
preprocessing and projection operators stay identical.

> [!TIP]
> - Ultralytics is the right tool when you are building exclusively with YOLO models
and want the smallest possible surface area.  
> - Raw ONNX Runtime is the right tool for zero-dependency constraints or highly unusual models.   
> - ml-pipes sits in between: the explicit control of raw ONNX with a reusable operator library that eliminates the repeated boilerplate.  


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

All examples auto-download their model and sample assets into
`examples/.example_assets/` on first run.

### Inference on files

| Example | Model | Task | Notable pipeline features |
|---|---|---|---|
| `run_detection.py` | any YOLOv8-compatible | detection | generic, bring your own model |
| `run_yolo8_onnx.py` | YOLOv8 | detection | baseline YOLO pipeline |
| `run_yolo11n_fp16.py` | YOLO11n FP16 | detection | `Cast` for FP16, letterbox resize |
| `run_rfdetr_nano.py` | DETR-style detector | detection | `Scale` for normalized boxes, softmax logits |
| `run_yolo11n_seg.py` | YOLO11n-seg | instance segmentation | prototype masks, `ReconstructMasks` + `FilterBy` |
| `run_maskrcnn.py` | Mask R-CNN int8 | instance segmentation | CNN family, NMS baked in, 28×28 RoI masks, BGR mean subtraction |
| `run_yolo8_batch.py` | YOLOv8 | batch detection | simple batch region usage |
| `run_yolo8_tile.py` | YOLOv8 | tiled detection | tile and merge style pipeline |

```bash
python examples/run_yolo8_onnx.py
python examples/run_yolo11n_seg.py
python examples/run_rfdetr_nano.py
python examples/run_maskrcnn.py
python examples/run_yolo8_tile.py
```

### Inspection, tracing, and benchmarking

| Example | Focus | Notes |
|---|---|---|
| `run_inspect.py` | step-by-step inspection | renders a successful pipeline run |
| `run_inspect_errors.py` | failed-run inspection | shows how inspection captures errors |
| `run_yolo8_tracing.py` | tracing | prints or captures per-step trace data |
| `run_yolo8_batch_benchmark.py` | single benchmark | benchmarks batch throughput on one pipeline |
| `examples/benchmarks/` | benchmark workflows | single runs, sweeps, axis sweeps, CLI benchmarking |

### Data preparation

| Example | Domain | Notes |
|---|---|---|
| `run_sms_spam_prepare.py` | tabular / text preparation | non-vision example built from data operators |

### Streaming and live inference

| Example | Model | Task | Notes |
|---|---|---|---|
| `streaming/run_yolo8_webcam.py` | YOLOv8 | live detection | reads from the default camera; press Q to quit |
| `run_yolo8_video.py` | YOLOv8 | video detection | sequential baseline; auto-downloads OpenCV's `vtest.avi` sample |
| `streaming/run_shibuya_counter.py` | CSRNet + detector | crowd counting pipeline |
| `streaming/run_shibuya_csrnet.py` | CSRNet | density-map based crowd estimation |
| `streaming/run_shibuya_rf.py` | DETR-style detector | streaming detector variant |

```bash
# Live webcam — press Q to quit
python examples/streaming/run_yolo8_webcam.py

# Video file — uses bundled sample, or pass --input clip.mp4
python examples/run_yolo8_video.py
python examples/run_yolo8_video.py --input clip.mp4 --output annotated.mp4
```

### Torch and domain handoff

| Example | Focus | Notes |
|---|---|---|
| `torch/run_mask2former_torch_postprocess.py` | Torch-heavy postprocess | keeps mask postprocessing in Torch |
| `torch/run_mask2former_numpy_postprocess.py` | NumPy handoff | converts back earlier and finishes in NumPy |

### Inference endpoint

| Example                  | Model | Task | Notes |
|--------------------------|---|---|---|
| `run_yolo8_endpoint.py` | YOLOv8 | HTTP detection endpoint | requires `pip install flask` |

```bash
# Terminal 1 — start the server
python examples/run_yolo8_endpoint.py

# Terminal 2 — send a test request (uses bundled sample, or pass --input photo.jpg)
python examples/run_yolo8_endpoint.py --call
python examples/run_yolo8_endpoint.py --call --input photo.jpg

# Or with curl
curl -s -X POST http://localhost:5000/detect \
     -H "Content-Type: application/octet-stream" \
     --data-binary @examples/.example_assets/coco_000000039769.jpg | python -m json.tool
```
