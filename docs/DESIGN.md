# Design

This document explains the conceptual model behind `ml-pipes`: what the
framework treats as primary, why composition is the core mechanism, and how
that differs from inheritance-heavy SDKs.

## What ml-pipes Treats As First-Class

`ml-pipes` starts from data flow. A pipeline is a sequence of operators that
push data forward: one operator receives a value, transforms it, and hands the
result to the next operator.

Within that flow, the operator is the unit of composition. Each operator
should own one meaningful step of logic and expose one clear boundary: a
readable input/output contract for that step. Larger behaviors stay flexible
because they are assembled by arranging operators rather than hidden inside
one object or method.

Once operators and their composition become the first-class concern, the
tooling naturally follows them.
Validation checks that operator boundaries connect, Inspection gives you a
built-in lineage view of what each step produced in one run, Tracing records
how a call moved through the pipeline, and Benchmark measures the same flow
across repeated runs.

> [!IMPORTANT]
> This makes the pipeline the primary artifact in `ml-pipes`, and the
> operator the unit of composition.

## Why ml-pipes Is Not Model-Centric

Models are important, but they are not the center of the framework. A model
call is one operator inside a larger data path that often also includes
loading inputs, preprocessing, runtime invocation, output decoding,
filtering, projection, evaluation, and export.

Treating the model as one operator rather than as a special framework mode
keeps the framework consistent across the whole workflow. The same composition
model and the same tooling work before the model call, around it, and after
it. The framework is therefore centered on the full data path rather than on
one special model abstraction.

In that sense, weights and biases matter because they affect how data is
transformed, not because the framework gives the model a special role of its
own.

## Why Composition Beats Inheritance

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

**Because explicitness matters, the obvious choice is composition.**

Composition keeps the workflow directly visible instead of distributing it
across an inheritance chain. A pipeline is a plain list of small, single-purpose
operators:

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

For example in this case, different models produce different outputs, but at some level of abstraction they
still produce boxes, scores, class indices, and optionally masks. The right
operators, applied in the right order, produce the right result regardless of
model family.

Switching from YOLOv8 to a DETR-style detector changes only a few lines (`Scale`
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

The same idea extends beyond the model itself. An ML workflow is usually a
sequence of transformations over artifacts: raw files to validated records,
records to features, features to predictions, predictions to metrics, or
results to downstream actions. That shape is why function-style code remains a
natural fit even when some stages still involve side effects at the edges.

## From Design To Architecture

This document has explained the conceptual model behind `ml-pipes`: what the
framework is built around and why those choices were made. To see how that
design is realized in the codebase, continue to
[ARCHITECTURE.md](ARCHITECTURE.md).

Taken together, the two docs answer different questions. `DESIGN.md` explains
why `ml-pipes` looks this way. `ARCHITECTURE.md` explains ownership
boundaries, runtime structure, and where changes should usually land.
