# Model Scaffolding Tutorial

This tutorial walks through bringing a new NumPy/ONNX model into
`ml-pipes`. The goal is not to hide the entire model family inside one large
operator. The goal is to build a first scaffold with explicit boundaries so
you can validate it, inspect intermediate values, and swap pieces as you learn
what the model actually exports.

The examples below use detection-style outputs, with a segmentation variant
where it matters. Start with built-in operators first. Add a local callable
only when the model has a real quirk that does not fit the existing operator
surface.

## Step 1 — Inspect The Model Outputs

Load the model in Netron or run a quick inspection with ONNX Runtime to find
the output tensor names and shapes:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
for output in session.get_outputs():
    print(output.name, output.shape)
```

This tells you what the graph actually publishes. Do not guess the output
names from another export or from a model family blog post. The scaffold
should be built from the concrete outputs in your file.

## Step 2 — Map Outputs Into The Tensor Registry

Use `Extract` to pull tensors by graph output name into the registry, renaming
them to semantic names:

```python
# Single output
Extract("output0", as_="preds")

# Multiple outputs
Extract("output0", "output1", as_=("preds", "protos"))

# Already semantically named (for example, some Mask R-CNN exports)
Extract("6568", "6570", "6572", "6887", as_=("boxes", "labels", "scores", "masks"))
```

The important part is that after `Extract`, the rest of the pipeline can talk
about semantic tensor roles like `boxes`, `scores`, or `protos` instead of raw
graph ids.

## Step 3 — Adapt The Raw Tensor Layout

Different model families export predictions in different layouts:

```python
# YOLO8/11: (1, features, N) -> squeeze batch dim, then transpose
Squeeze("preds"),          # (1, 116, N) -> (116, N)
Transpose("preds"),        # (116, N) -> (N, 116)

# YOLOv5: (1, N, features) -> no transpose needed
Squeeze("preds"),          # (1, 25200, 117) -> (25200, 117)
```

Keep this as its own explicit step. Layout fixes are one of the most common
places where a new model scaffold goes wrong, and they are much easier to
debug when they are visible in the pipeline.

## Step 4 — Slice Out Semantic Tensors

Once the raw prediction tensor is in the right layout, split it into the
semantic pieces the rest of the pipeline expects:

```python
Slice("preds", slice(None, 4), as_="boxes"),          # (N, 4)
Slice("preds", slice(4, -32), as_="class_scores"),    # (N, 80)
Slice("preds", slice(-32, None), as_="mask_coeffs"),  # (N, 32)
```

From here on, the pipeline should read like the problem domain: boxes, class
scores, mask coefficients, prototypes, and so on.

## Step 5 — Handle Model-Specific Coordinate Formats

Some models emit normalized coordinates that need scaling to pixel space
before NMS:

```python
# DETR-style model: normalized cxcywh -> pixel cxcywh
Scale("boxes", by=(input_w, input_h, input_w, input_h)),
ConvertBoxFormat("boxes", from_="cxcywh", to="xyxy"),
```

If the model has a real quirk that does not fit the existing operators, use a
plain local callable instead of forcing the whole scaffold into one custom
class:

```python
from ml_pipes import Pipeline, TensorRegistry


def _my_model_quirk(registry: TensorRegistry) -> TensorRegistry:
    # Handle whatever the model does unusually.
    registry["boxes"] = ...
    return registry


pipeline = Pipeline([
    ...,
    _my_model_quirk,
    ...,
])
```

Keep that callable small and specific. The scaffold should stay mostly built
from ordinary operators.

## Step 6 — Apply NMS And Project Back

Store the resize transform before inference so it can be recalled later for
projection:

```python
Resize((640, 640)),
Store("resize_transform", index=1),
Pick(0),
...,
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

This is where the scaffold becomes a real end-to-end model boundary: model
space back to source-image space, then into the framework's output objects.

## Final Detection Scaffold

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

## What To Do Next

- Run `pipeline.validate()` once the scaffold is assembled.
- Use inspection to confirm each intermediate tensor has the shape and meaning
  you expect.
- Keep model-specific quirks local and small; reuse built-in operators
  everywhere else.
- If you need to define a new operator, see [OPERATORS.md](OPERATORS.md).
- For validation rules and `resolve_contract(...)`, see
  [VALIDATION.md](VALIDATION.md).
- For composition patterns around the scaffold, see
  [COMPOSITION.md](COMPOSITION.md).
