# ml-pipes

Composable ONNX inference pipelines. Build detection, segmentation, and
classification pipelines by chaining small reusable operators — explicit,
individually testable, and model-agnostic.

## Install

```bash
pip install -e .
```

## Quick start

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
print(detections.boxes, detections.scores, detections.classes)
```

Switching to a different model family (RF-DETR, Mask R-CNN, YOLO11) means
changing the operators between `Infer` and `ToDetections` — preprocessing and
projection operators are shared and unchanged. See
[Building a pipeline for a custom model](#building-a-pipeline-for-a-custom-model).

## Design principle

Most inference SDKs are built around inheritance: a `Detector` base class with
a `YoloDetector` subclass, a `Segmentor` base class with a `MaskRCNNSegmentor`
subclass. This feels natural at first, but it couples the pipeline logic to a
specific model family. Adding a new model means subclassing; changing
postprocessing means overriding methods; reusing a preprocessing step across
model families means fighting the hierarchy.

**ml-pipes takes the opposite approach: composition.**

A pipeline is a plain list of small, single-purpose operators. Each operator
does one thing and knows nothing about the model, task, or runtime. Different
models produce different outputs — but at some level of abstraction they all
produce boxes, scores, class indices, and optionally masks. The right operators
applied in the right order produce the right result regardless of model family.

```
image file
  → decode → resize → normalize             # preprocessing, model-agnostic
  → infer                                   # the only model-specific step
  → select → squeeze → transpose            # adapt raw output to registry
  → slice → argmax → gather_scores          # extract semantic tensors
  → convert_box_format → nms                # detection-specific logic
  → recall transform → project_boxes        # postprocessing, model-agnostic
  → to_detections                           # output type
```

The same operators appear in every detection pipeline. Switching from YOLOv8
to RF-DETR changes three lines (a `Scale` for normalized boxes and different
softmax/argmax handling) — the rest is identical.

### Why function-style coding is natural for inference

A neural network is, fundamentally, a function: it maps an input tensor to
output tensors. The entire inference pipeline — decode, resize, normalize,
infer, postprocess — is also a function: it maps an image to a structured
prediction. Every step in between is a function too. The problem is shaped like
function composition from top to bottom, so the code should be too.

Object-oriented inheritance fights this shape. A `Detector` class with a
`predict` method hides the transformation sequence inside method calls and
inherited overrides. To understand what happens to the data you trace through
multiple layers of the class hierarchy. The interesting computation — what
actually happens to the tensors — is scattered.

A pipeline list makes the data transformation the primary artifact. Reading the
pipeline top to bottom tells you exactly what happens to the data, in order,
without indirection. There is no hidden state between steps: each operator
receives a value, returns a value, and has no memory of previous calls. This
mirrors how you reason about inference — "resize the image, normalize it, run
the model, squeeze the batch dimension, threshold by confidence" — and it means
that reasoning is directly visible in the code rather than distributed across a
class hierarchy.

This also means the pipeline is inspectable and debuggable at every boundary.
Inserting a `print` function or a logging step at any position in the list
shows you the exact value flowing through at that point. There are no private
fields to dig into, no method override chain to follow.

## Advantages

**Portability across model families.**
A new model is a new pipeline, not a new class. YOLOv8, YOLO11, RF-DETR,
Mask R-CNN, and future models all run through the same operator library.

**Explicit, readable pipelines.**
The pipeline list is a complete description of what happens to the data.
There are no hidden method overrides, no inherited behaviour to trace. Reading
the pipeline list tells you everything.

**Reusability without abstraction overhead.**
`Normalize`, `ConvertBoxFormat`, `ProjectBoxes` are used across every model.
They are not tied to any base class. There is no cost to reuse them — just add
them to a pipeline.

**Model-specific logic stays at the boundary.**
When a model has quirky output (e.g. RF-DETR's normalized boxes), the
adaptation is a single operator or plain callable in the pipeline. It doesn't
pollute shared infrastructure.

**Testability.**
Each operator is a plain Python object. Testing `NMS` means calling
`NMS()(registry)` with a hand-crafted `TensorRegistry`. No mocking, no
subclass setup.

**Optional early validation.**
`Pipeline(validate_on_init=True)` checks type contracts at construction time
using Python type annotations, catching operator mismatches before any data
flows through.

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

Maximum control, but every part is hand-written. Switching to RF-DETR means
rewriting the entire preprocessing and postprocessing block from scratch — the
confidence filter, box conversion, NMS, and projection are all repeated. Nothing
here is reusable across model families.

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
named and individually testable. Switching to RF-DETR changes three operators
(`Scale` for normalised boxes, `Softmax` before `ArgMax`) — all preprocessing
and projection operators are identical and unchanged.

### Summary

| | Ultralytics | Raw ONNX Runtime | ml-pipes |
|---|---|---|---|
| Model scope | YOLO family | Any ONNX | Any ONNX |
| Pipeline visibility | Opaque | Fully explicit | Fully explicit |
| Operator reuse across models | Not applicable | Manual copy-paste | Shared operator library |
| Custom postprocessing | Subclass or post-hoc | Full freedom | Insert any callable |
| Testability | Integration tests only | Unit tests with boilerplate | Unit tests on individual operators |
| Brevity | High | Low | Medium |
| ONNX-native | Export step required | Yes | Yes |

Ultralytics is the right tool when you are building exclusively with YOLO models
and want the smallest possible surface area. Raw ONNX Runtime is the right tool
for zero-dependency constraints or highly unusual models. ml-pipes sits in
between: the explicit control of raw ONNX with a reusable operator library that
eliminates the repeated boilerplate.

## Building a pipeline for a custom model

This is the general process for adding support for a new ONNX model.

### Step 1 — Inspect the model outputs

Load the model in Netron or run a quick inspection with ONNX Runtime to find
the output tensor names and shapes:

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
for o in session.get_outputs():
    print(o.name, o.shape)
```

### Step 2 — Map outputs to the TensorRegistry

Use `Extract` to pull tensors by their graph output names into the registry, renaming to
semantic names:

```python
# Single output
Extract("output0", as_="preds")

# Multiple outputs
Extract("output0", "output1", as_=("preds", "protos"))

# Already semantically named (e.g. some Mask R-CNN exports)
Extract("6568", "6570", "6572", "6887", as_=("boxes", "labels", "scores", "masks"))
```

### Step 3 — Adapt the raw tensor layout

Different model families export predictions in different layouts:

```python
# YOLO8/11: (1, features, N) — squeeze batch dim, then transpose
Squeeze("preds"),          # (1, 116, N) → (116, N)
Transpose("preds"),        # (116, N) → (N, 116)

# YOLOv5: (1, N, features) — no transpose needed
Squeeze("preds"),          # (1, 25200, 117) → (25200, 117)
```

### Step 4 — Slice out semantic tensors

```python
Slice("preds", slice(None, 4), as_="boxes"),          # (N, 4)
Slice("preds", slice(4, -32), as_="class_scores"),    # (N, 80)
Slice("preds", slice(-32, None), as_="mask_coeffs"),  # (N, 32)
```

### Step 5 — Handle model-specific coordinate formats

Some models output normalized coordinates that need scaling to pixel space
before NMS:

```python
# RF-DETR: normalized cxcywh → pixel cxcywh
Scale("boxes", by=(input_w, input_h, input_w, input_h)),
ConvertBoxFormat("boxes", from_="cxcywh", to="xyxy"),
```

For truly model-specific logic that doesn't fit any operator, use a plain
callable:

```python
def _my_model_quirk(registry: TensorRegistry) -> TensorRegistry:
    # handle whatever the model does unusually
    registry["boxes"] = ...
    return registry

pipeline = Pipeline([
    ...,
    _my_model_quirk,
    ...
])
```

### Step 6 — NMS and projection

Store the resize transform before inference so it can be recalled for
projection:

```python
Resize((640, 640)),
Store("resize_transform", index=1),
Pick(0),
...                                     # inference and postprocessing
NMS(),
Recall("resize_transform"),
ProjectBoxes(),
ToDetections(),
```

For segmentation, project boxes first, then masks:

```python
NMS(kept_as="kept"),
FilterBy("mask_coeffs", "kept"),
ReconstructMasks("mask_coeffs", "protos", as_="masks"),
Recall("resize_transform"),
ProjectBoxes(),
Recall("resize_transform"),
ProjectMasks("masks", mask_threshold=0.5),
ToSegmentations(),
```

### Complete example

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
    Infer("model.onnx"),
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

## Writing a custom operator

An operator is any Python callable. For stateless, single-use logic, a plain
function is enough:

```python
def drop_background_class(registry: TensorRegistry) -> TensorRegistry:
    # remove class 0 (background) detections
    kept = registry["classes"] != 0
    for key in ("boxes", "scores", "classes"):
        registry[key] = registry[key][kept]
    return registry
```

For reusable, parameterised operators, use a class with `__call__`:

```python
from ml_pipes import TensorRegistry

class DropClass:
    """Removes all detections whose class index equals `class_id`."""

    def __init__(self, class_id: int):
        self.class_id = class_id

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        kept = registry["classes"] != self.class_id
        for key in ("boxes", "scores", "classes"):
            registry[key] = registry[key][kept]
        return registry
```

Used in a pipeline:

```python
Pipeline([
    ...,
    NMS(),
    DropClass(class_id=0),    # remove background before projecting
    Recall("resize_transform"),
    ProjectBoxes(),
    ToDetections(),
])
```

### Accessing the context from a custom operator

If your operator needs a value stored via `Store`, annotate the second
positional parameter with its type. After a matching `Recall`, the pipeline
will pass the stored value automatically:

```python
from ml_pipes import TensorRegistry, ResizeTransform

def my_projection(registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
    orig_h, orig_w = transform.original_shape
    # ... custom projection logic
    return registry
```

```python
Pipeline([
    ...,
    Recall("resize_transform"),
    my_projection,              # receives (registry, transform)
    ToDetections(),
])
```

### Contract validation

Add type annotations to `__call__` and `Pipeline(validate_on_init=True)` will
verify that each operator's input type is compatible with the previous
operator's output type at construction time:

```python
class DropClass:
    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        ...
```

## Extending the pipeline

### Wrapping a pipeline

If you run the same model repeatedly, wrap the pipeline in a function or class
that owns the `Pipeline` instance:

```python
class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self._pipeline = Pipeline([
            Decode(),
            Resize((640, 640)),
            Store("resize_transform", index=1),
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
```

### Chaining pipelines

TBA.

## Examples

All examples auto-download their model and sample assets into
`.example_assets/` on first run.

### Inference on files

| Example | Model | Task | Notable pipeline features |
|---|---|---|---|
| `run_detection.py` | any YOLOv8-compatible | detection | generic, bring your own model |
| `run_yolo8n_onnx.py` | YOLOv8n | detection | baseline YOLO pipeline |
| `run_yolo11n_onnx_fp16.py` | YOLO11n FP16 | detection | `Cast` for FP16, letterbox resize |
| `run_rfdetr_nano_onnx.py` | RF-DETR nano | detection | `Scale` for normalized boxes, softmax logits |
| `run_yolo11n_seg_onnx.py` | YOLO11n-seg | instance segmentation | prototype masks, `ReconstructMasks` + `FilterBy` |
| `run_maskrcnn_onnx.py` | Mask R-CNN int8 | instance segmentation | CNN family, NMS baked in, 28×28 RoI masks, BGR mean subtraction |

```bash
python examples/run_yolo8n_onnx.py
python examples/run_yolo11n_seg_onnx.py
python examples/run_rfdetr_nano_onnx.py
python examples/run_maskrcnn_onnx.py
```

### Live and video inference

| Example | Model | Task | Notes |
|---|---|---|---|
| `run_webcam_yolo8n_onnx.py` | YOLOv8n | live detection | reads from the default camera; press Q to quit |
| `run_video_yolo8n_onnx.py` | YOLOv8n | video detection | sequential baseline; auto-downloads OpenCV's `vtest.avi` sample |

```bash
# Live webcam — press Q to quit
python examples/run_webcam_yolo8n_onnx.py

# Video file — uses bundled sample, or pass --input clip.mp4
python examples/run_video_yolo8n_onnx.py
python examples/run_video_yolo8n_onnx.py --input clip.mp4 --output annotated.mp4
```

### Inference endpoint

| Example | Model | Task | Notes |
|---|---|---|---|
| `serve_yolo8n_onnx.py` | YOLOv8n | HTTP detection endpoint | requires `pip install flask` |

```bash
# Terminal 1 — start the server
python examples/serve_yolo8n_onnx.py

# Terminal 2 — send a test request (uses bundled sample, or pass --input clip.mp4)
python examples/serve_yolo8n_onnx.py --call
python examples/serve_yolo8n_onnx.py --call --input photo.jpg

# Or with curl
curl -s -X POST http://localhost:5000/detect \
     -H "Content-Type: application/octet-stream" \
     --data-binary @.example_assets/coco_000000039769.jpg | python -m json.tool
```
