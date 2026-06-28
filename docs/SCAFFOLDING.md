# Model Scaffolding Tutorial

This guide shows how to create a model scaffold with `ml-pipes`.

A scaffold is the smallest useful pipeline around model execution. Instead of
treating the model call as one opaque block, you turn the integration into
named steps that are easier to validate, inspect, debug, benchmark, adapt,
and reuse inside a larger workflow or application.

`ml-pipes` does not require a specific runtime. You can use the core
`ml-pipes` steps with `Infer(...)` for ONNX Runtime, the `ml_pipes.torch`
package for Torch-native steps, or your own callable/operator around any
library that can load weights and run the model.

The walkthrough below uses a simple detection-style scaffold so the main
pattern stays easy to see, but the same scaffolding approach applies to any
model you can load and run from Python.

## Step 0 — Check The Model Outputs

Before you design the scaffold, check what the model actually returns. Do not
guess this from another export or from a model-family blog post.

Inspect the real values your runtime exposes: names, keys, shapes, dtypes, and
layout.

For ONNX Runtime, a quick inspection looks like this:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
for output in session.get_outputs():
    print(output.name, output.shape)
```

For other runtimes, inspect whatever the model actually returns: tuple
position, dict key, tensor shape, and dtype. The scaffold should be built from
the concrete outputs in your runtime, not from an assumed interface.

## Step 1 — Clarify The Model Boundary

Once you know what the model actually returns, decide what the model step
should own and what should stay outside it.

In most cases, the ideal model boundary is narrow:

- preprocessing stays outside the model step
- the model step receives the prepared value the runtime expects
- the model step returns raw outputs or only lightly adapted outputs
- postprocessing stays outside the model step

That keeps responsibilities clear. Resize, normalize, color conversion,
batching, layout adaptation, score extraction, candidate filtering,
suppression, and projection are easier to inspect and change when they stay as
ordinary pipeline steps instead of being buried inside the model wrapper.

Different models may need different runtimes or different postprocessing, but
that overall structure stays the same. Keep the model call as one explicit step
inside a larger flow.

For example:

```python
from ml_pipes import Infer
from ml_pipes.torch import ToNumpyRegistry, ToTorch, TorchInfer

# ONNX Runtime
Infer("model.onnx")

# Torch-native model
ToTorch(device="cpu")
TorchInfer(model, input_layout="NCHW")
ToNumpyRegistry()

# Any other library
RunMyModel()
```

Use the built-in runtime operators when they fit. If your model runs through a
different library, wrap that call in a plain callable or operator so the model
itself is still one visible step in the pipeline.

## Step 2 — Prepare The Model Inputs

Before the model step, most scaffolds need a small preprocessing block.

Typical input preparation includes:

- decode or load the input into the value your pipeline works on
- resize, pad, or crop to the model's expected shape
- convert color space, layout, or dtype when needed
- normalize or scale values
- add a batch dimension only if the runtime expects it
- store any transform metadata you will need later for projection

For example:

```python
Decode(),
Resize((640, 640)),
Store("resize_transform", source=1),
Pick(0),
Normalize(),
```

Keep this separate from the model call. It defines the model input boundary
and makes input mistakes much easier to inspect and fix.

## Step 3 — Map Outputs Into A TensorRegistry

If you use `Infer(...)`, the raw result is `RuntimeOutputs`: the model outputs
exactly as exposed by the runtime, together with their names.

In `ml-pipes`, a `TensorRegistry` is just a named store of tensors that moves
from step to step through the pipeline.

The rest of the scaffold is easier to reason about once that flowing value
uses meaningful names like `boxes`, `scores`, or `logits`.

With ONNX Runtime outputs, use `Extract` to pull tensors by graph output name
into the registry:

```python
# Single output
Extract("output0", as_="preds")

# Multiple outputs
Extract("pred_boxes", "logits", as_=("boxes", "logits"))
```

With another runtime, do the same mapping in a small callable or operator:

```python
from ml_pipes import TensorRegistry


def outputs_to_registry(outputs) -> TensorRegistry:
    return TensorRegistry({
        "boxes": outputs["boxes"],
        "logits": outputs["logits"],
    })
```

If you write your own model step, you can either return something equivalent
to `RuntimeOutputs` and reuse `Extract(...)`, or skip that shape entirely and
return a `TensorRegistry` directly.

The important part is that after this mapping step, the rest of the pipeline
can talk about values like `boxes`, `scores`, or `logits` instead of raw graph
output ids.

## Step 4 — Adapt The Raw Tensor Layout

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

## Step 5 — Slice Out The Parts You Need

Once the raw prediction tensor is in the right layout, split it into the
pieces the rest of the pipeline expects:

```python
Slice("preds", slice(None, 4), as_="boxes"),         # (N, 4)
Slice("preds", slice(4, None), as_="class_scores"),  # (N, C)
```

If the model emits extra tensors, split them out here the same way.

From here on, the pipeline should read like the problem domain: boxes, class
scores, labels, keypoints, or whatever the later steps need.

## Optional Step — Handle Model Quirks

Most model quirks still fit the generic tensor/registry steps that already
exist in `ml-pipes`, so try those first.

For example, Mask R-CNN exports 1-indexed labels, so the fix is just:

```python
MapTensor("labels", fn=lambda t: t.astype("int32") - 1, as_="classes")
```

For a more involved example built from generic steps, see the examples at the
end of this guide.

If the quirk still does not fit the existing steps, use a small local
callable instead of forcing the whole scaffold into one custom class:

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
from ordinary `ml-pipes` steps. If that logic stops being local and becomes
reusable, turn it into a real operator; see [OPERATORS.md](OPERATORS.md).

Some models will not need this step at all. Keep it optional and local.

## Step 6 — Handle Model-Specific Coordinate Formats

Some models emit coordinates that still need cleanup in model space before you
filter candidates or project anything back to the source image.

That can mean denormalizing coordinates, changing box formats, or both:

```python
# DETR-style model: normalized cxcywh -> pixel cxcywh
Scale("boxes", by=(input_w, input_h, input_w, input_h)),
ConvertBoxFormat("boxes", from_="cxcywh", to="xyxy"),
```

Keep this as its own step. It is still part of adapting the model outputs in
model space. Projection back to the source image happens later.

## Step 7 — Filter And Reduce Candidates

Before you project anything back to the source image or do heavier downstream
work, reduce the number of candidates.

As a rule of thumb, go from least expensive to most expensive:

- apply simple per-candidate filters first, such as score thresholds, class
  filters, or simple top-k reduction
- then apply pairwise suppression or merge, such as `NMS()` or `NMM()`
- if later tensors need to stay aligned, apply the same kept indices to them
- leave heavier postprocessing and projection for later

For a simple detection scaffold, that might look like this:

```python
FilterTensorsByScore("boxes", "classes", score="scores", min_score=0.25),
NMS(),
```

In a very small pipeline, `NMS(conf_threshold=...)` may already be enough by
itself. Add explicit early filters when you want that reduction to happen
before later steps.

The exact operators can change by model family. The important point is the
ordering: cheap filters first, heavier work later.

## Step 8 — Project Back To The Source Image

Store the resize transform before inference so you can use it later to map the
results back to the original image:

```python
Resize((640, 640)),
Store("resize_transform", source=1),
Pick(0),
...,
NMS(),
Recall("resize_transform"),
ProjectBoxes(),
ToDetections(),
```

Here `Store(...)` saves the resize metadata, and `Recall(...)` brings it back
later when `ProjectBoxes()` needs to convert coordinates back to the source
image. `Pick(0)` keeps the resized image flowing forward after `Resize(...)`
returns both the resized value and the transform metadata.

This is where the scaffold turns raw model outputs into final values your
application can use: first back into source-image coordinates, then into
objects such as `Detections`.

## Final Detection Scaffold

The example below uses ONNX Runtime for the model step. The same overall
scaffold still works if you replace that step with a Torch-based step or a
callable that runs another library.

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
    Infer("model.onnx"),
    Extract("output0", as_="preds"),
    Squeeze("preds"),
    Transpose("preds"),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="class_scores"),
    ArgMax("class_scores", as_="classes"),
    GatherScores("class_scores", "classes", as_="scores"),
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

### Validate The Pipeline

Run `pipeline.validate()` once the scaffold is assembled and again whenever
you change what a step takes in or returns. That catches step-to-step
mismatches before they turn into runtime bugs. For validation rules and
advanced hooks such as `resolve_contract(...)`, see [VALIDATION.md](VALIDATION.md).

### Debug With Inspection

Use `pipeline.inspect(sample_input)` to confirm that each step produces the
shape and meaning you expect. If the scaffold is wrong, inspection usually
shows the first bad step immediately. For inspection examples and output
formatting, see [OPERATORS.md](OPERATORS.md) and
[run_inspect.py](../examples/run_inspect.py).

### Keep The Scaffold Composable

Keep the actual model step narrow, keep quirks local, and keep the rest of the
flow explicit. That makes the scaffold easier to swap, extend, and reuse
across model variants. For composition patterns, see
[COMPOSITION.md](COMPOSITION.md).

## More Examples

For segmentation variants or heavier postprocessing, see:

- [run_maskrcnn.py](../examples/run_maskrcnn.py) for RoI-mask segmentation with a small label remap
- [run_yolo11n_seg.py](../examples/run_yolo11n_seg.py) for a YOLO segmentation scaffold with prototype masks
- [run_mask2former_numpy_postprocess.py](../examples/torch/run_mask2former_numpy_postprocess.py) for a more involved NumPy postprocess built from generic steps
