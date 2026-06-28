# Design

This document explains why `ml-pipes` is built around explicit operator
composition, why that fits ML workflows, and how that compares with a few
common alternatives.

## Overview

`ml-pipes` starts from data flow. A pipeline is a sequence of operator
boundaries that push data forward: one step receives a value, transforms it,
and hands the result to the next step. The value might be an image, a tensor
registry, a batch, a record, or any other payload, but the main thing the
framework cares about is how that data moves and changes.

Once data becomes the first-class concern, the tooling naturally follows it.
Validation checks that operator boundaries connect, Inspection gives you a
built-in lineage view of what each step produced in one run, Tracing records
how a call moved through the pipeline, and Benchmark measures the same flow
across repeated runs.

Even models are not first-class in that sense. A model call is just one
operator in a larger data path; weights and biases matter because they affect
how data is transformed, not because the framework treats the model itself as
the center of the system.

## Composition Over Inheritance

Many inference SDKs are built around inheritance. That feels natural at first,
but in a mature codebase the prediction path often ends up spread across
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
models produce different outputs, but at some level of abstraction they still
produce boxes, scores, class indices, and optionally masks. The right
operators, applied in the right order, produce the right result regardless of
model family. Most of the pipeline stays the same; only the model-specific
decoding changes.

```python
# YOLOv8n
Pipeline([
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
```

Switching from YOLOv8 to a DETR-style detector changes a few lines (`Scale`
for normalized boxes and different softmax/argmax handling); the rest stays
explicit, reusable, and testable:

```python
# DETR-style detector: only the section between Infer and NMS changes
Pipeline([
    Decode(),
    ...,
    Infer("detr_nano.onnx"),
    Extract("pred_boxes", "logits", as_=("boxes", "logits")),
    Squeeze("boxes"),
    Squeeze("logits"),
    Softmax("logits"),
    ArgMax("logits", as_="classes"),
    GatherScores("logits", "classes", as_="scores"),
    Scale("boxes", by=(640, 640, 640, 640)),
    ConvertBoxFormat(from_="cxcywh"),
    ...,
])
```

## Why Function-Style Coding Fits ML Workflows

In many ordinary applications, most of the code is side effects with
a little data mutation around them. In ML applications, the ratio is usually
reversed: most of the work is data mutation, with side effects mostly at the
edges when you load inputs, call runtimes, or save outputs.

Once you treat ML applications as transformations over data and artifacts, function-style
code becomes the natural fit. Loading maps files to records. Cleaning maps
records to cleaner records. Feature builders map records to tensors. Batchers
map examples to batches. Models map tensors to tensors. Evaluators map
predictions and labels to metrics. Export steps map internal results to files,
tables, or API responses.

A neural network is just one compact example:

```python
import torch.nn as nn

# A network is a function: input tensor -> output tensor.
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
> This also means the pipeline is inspectable and debuggable at every
> boundary. Inserting a `print` function or a logging step at any position in
> the list shows you the exact value flowing through at that point. There are
> no private fields to dig into and no method override chain to follow.

For the operator model and composition semantics behind this style, see
[OPERATORS.md](OPERATORS.md) and [COMPOSITION.md](COMPOSITION.md).

## Comparison With Other Approaches

Below is the same task, YOLOv8n object detection on a single image,
implemented with three different approaches.

### Ultralytics

Ultralytics ships a high-level API tightly coupled to the YOLO model family:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict("image.jpg", conf=0.25, iou=0.45)

for r in results:
    print(r.boxes.xyxy)  # boxes in original image space
    print(r.boxes.cls)   # class indices
    print(r.boxes.conf)  # confidence scores
```

Three lines from model load to result. The entire preprocessing, inference,
and postprocessing pipeline runs inside `predict`. This is the right choice if
you are building exclusively with YOLO models and the default postprocessing
meets your needs.

The cost is opacity and lock-in. You cannot swap a preprocessing step, insert
a custom tensor operation, or reuse any of the internal logic with a non-YOLO
model. The pipeline is not a list of steps you control; it is a method you
call.

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

# Adapt layout: (1, 84, 8400) -> (8400, 84)
preds = preds.squeeze().T
boxes_cxcywh = preds[:, :4]
class_scores = preds[:, 4:]
classes = class_scores.argmax(axis=1)
scores = class_scores[np.arange(len(classes)), classes]

# Confidence filter
keep = scores >= 0.25
boxes_cxcywh, scores, classes = boxes_cxcywh[keep], scores[keep], classes[keep]

# cxcywh -> xyxy
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
```

More explicit than Ultralytics and more structured than raw ONNX. Every step
is named and individually testable. Switching to a DETR-style model changes a
few operators (`Scale` for normalized boxes and `Softmax` before `ArgMax`);
preprocessing and projection operators stay identical.

> [!TIP]
> Ultralytics is the right tool when you are building exclusively with YOLO
> models and want the smallest possible surface area.
>
> Raw ONNX Runtime is the right tool for zero-dependency constraints or highly
> unusual models.
>
> `ml-pipes` sits in between: the explicit control of raw ONNX with a reusable
> operator library that eliminates repeated boilerplate.
