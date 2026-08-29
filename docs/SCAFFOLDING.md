# Model Scaffolding Tutorial

This guide shows how to scaffold a model integration with `ml-pipes`. Before you start, read [README.md](../README.md) and
[COMPOSITION.md](COMPOSITION.md).

A scaffold is the smallest useful pipeline around model execution. In ML
terms, it is the explicit path from raw input to final predictions:
preprocessing, model execution, output decoding, filtering, and projection.

At a high level, scaffolding usually follows this flow:

- inspect the model interface
- choose the boundary around the model step
- prepare inputs
- run the model
- rename and decode outputs
- filter and finalize predictions

`ml-pipes` fits this naturally because each part of that flow can stay as one
or more explicit pipeline steps. It also does not require a specific runtime:
use `Infer(...)` for ONNX Runtime, `ml_pipes.torch` for Torch-native steps, or
your own callable/operator around another library.

The walkthrough below uses a simple detection-style scaffold so the main
pattern stays easy to see, but the same approach applies to any model you can
load and run from Python.

## Step 0 — Inspect The Model Interface

Before you design the scaffold, check what the model actually expects and what
it returns. Do not guess this from another export or from a model-family blog
post.

Inspect the real input names, shapes, dtypes, layouts, and output names,
shapes, dtypes, and layouts.

For ONNX Runtime, a quick inspection looks like this:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
for node in session.get_inputs():
    print("input", node.name, node.shape, node.type)
for node in session.get_outputs():
    print("output", node.name, node.shape, node.type)
```

For other runtimes, inspect whatever the model actually consumes and returns:
tuple position, dict key, tensor shape, dtype, and layout. This is the ground
truth for the scaffold.

## Step 1 — Clarify The Model Boundary

Once you know the model interface, decide what belongs in preprocessing, what
belongs in the model step itself, and what belongs in postprocessing.

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

In `ml-pipes`, that model step can be as small as:

```python
# ONNX Runtime
Infer("model.onnx")

# Torch-native model
TorchInfer(model, input_layout="NCHW")

# Any other runtime
RunMyModel()
```

Use the built-in runtime operators when they fit. If your model runs through a
different library, wrap that call in a small callable or operator so the model
itself is still one visible step in the pipeline. For Torch-specific
boundaries, see [packages/torch/docs/README.md](../packages/torch/docs/README.md).

## Step 2 — Prepare The Model Inputs

Before the model step, build the smallest preprocessing block that gets the
raw input into the form the model expects.

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

## Step 3 — Map Raw Outputs To Named Tensors

After inference, the first postprocessing step is usually to give the raw
outputs semantic names.

This is the bridge from runtime-specific output ids to task-specific tensors
like `boxes`, `scores`, or `logits`.

With ONNX Runtime outputs, use `Extract` to pull tensors by graph output name
into named tensors:

```python
# Single output
Extract("output0", as_="preds")

# Multiple outputs
Extract("pred_boxes", "logits", as_=("boxes", "logits"))
```

In `ml-pipes`, the named tensor collection that flows through later steps is a
`TensorRegistry`. If you use `Infer(...)`, the raw runtime return is
`RuntimeOutputs`, and `Extract(...)` is the normal adapter from that runtime
shape into named tensors.

If you run the model through your own callable, return the same named tensors
directly or build a small adapter around that runtime. The important part is
that after this step, the rest of the scaffold can talk about semantic values
instead of raw graph ids. For more operator patterns, see
[OPERATORS.md](OPERATORS.md).

## Step 4 — Adapt The Raw Output Layout

Many models still need some small tensor-shape fixes such as squeeze or
transpose before the main postprocessing starts.

For example:

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

## Step 5 — Split Predictions Into Semantic Tensors

Once the raw prediction tensor is in the right layout, split it into the
pieces the rest of the pipeline expects:

```python
Slice("preds", slice(None, 4), as_="boxes"),         # (N, 4)
Slice("preds", slice(4, None), as_="class_scores"),  # (N, C)
```

From here on, the pipeline should read like the problem domain: boxes, class
scores, labels, keypoints, or whatever the later steps need. This is where the
scaffold starts to look like your task instead of your runtime export.

## Optional Step — Handle Model Quirks

Most model quirks still fit the generic tensor operations that already exist
in `ml-pipes`, so try those first.

For example, Mask R-CNN exports 1-indexed labels, so the fix is just:

```python
MapTensor("labels", fn=lambda t: t.astype("int32") - 1, as_="classes")
```

If the quirk still does not fit the existing steps, keep it in a small local
callable or operator instead of folding it into the entire scaffold. If that
logic becomes reusable, turn it into a real operator; see
[OPERATORS.md](OPERATORS.md). For more involved postprocessing examples, see
the examples at the end of this guide.

Some models will not need this step at all. Keep it optional and local.

## Step 6 — Standardize Model-Specific Coordinates

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

Before you project anything back to the source image, reduce the number of
candidates.

As a rule of thumb, go from least expensive to most expensive:

- apply simple per-candidate filters first, such as class filters, area
  filters, simple top-k reduction, or other task-specific pruning
- then apply pairwise suppression or merge, such as `NMS()` or `NMM()`
- when aligned tensors must follow the same reduction, use either boolean-mask
  filtering or index slicing

For a simple detection scaffold, that might look like this when you want to
drop very small objects before `NMS()`:

```python
FilterTensors(
    "boxes",
    "scores",
    "classes",
    by="boxes",
    predicate=lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) >= 16 * 16,
),
NMS(),
```

If score thresholding is all you need, `NMS(conf_threshold=...)` may already
be enough by itself.

For more filtering and selection operators, start with
[PACKAGES.md](PACKAGES.md) and then check the relevant package docs for your
domain.

The exact operators can change by model family. The important point is the
ordering: cheap filters first, heavier work later.

## Step 8 — Project Back To The Source Image

Reuse the resize or pad metadata from preprocessing when you map predictions
back to the original image:

```python
Resize((640, 640)),
Store("resize_transform", source=1),
Pick(0),
...,
NMS(),
Recall("resize_transform"),
ProjectBoxes(),
```

In `ml-pipes`, `Store(...)` and `Recall(...)` are the usual way to carry that
metadata through the pipeline.

This is where the scaffold turns raw model outputs into source-image
coordinates and leaves the named prediction tensors available for rendering,
logging, or application-specific export.

## Putting The Flow Together

Read the scaffold from top to bottom as: prepare inputs, run the model, name
and decode the outputs, filter candidates, then project back to the source
image.

```python
pipeline = Pipeline([
    # Prepare inputs
    Decode(),
    Resize((640, 640)),
    Store("resize_transform", source=1),
    Pick(0),
    Normalize(),

    # Run model
    Infer("model.onnx"),

    # General model postprocessing
    Extract("output0", as_="preds"),
    Squeeze("preds"),
    Transpose("preds"),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="class_scores"),
    ArgMax("class_scores", as_="classes"),
    GatherRows("class_scores", "classes", as_="scores"),

    # Vision-specific postprocessing
    ConvertBoxFormat(from_="cxcywh"),
    NMS(),
    Recall("resize_transform"),
    ProjectBoxes(),
])
```

The exact operators can change by model family, but the scaffold shape usually
stays the same.

## What To Do Next

- Run `pipeline.validate()` as you shape the scaffold. For validation rules and
  advanced hooks such as `resolve_contract(...)`, see
  [VALIDATION.md](VALIDATION.md).
- Use `pipeline.inspect(sample_input)` to look at intermediate tensors and find
  the first wrong step.
- If local logic grows into reusable pipeline code, see
  [OPERATORS.md](OPERATORS.md) and [COMPOSITION.md](COMPOSITION.md).

## More Examples

See [INSPECTION.md](INSPECTION.md) for usage and
[run_inspect.py](../examples/run_inspect.py) for a runnable example.

For segmentation variants or heavier postprocessing, see:

- [run_maskrcnn.py](../examples/run_maskrcnn.py) for RoI-mask segmentation with a small label remap
- [run_yolo11n_seg.py](../examples/run_yolo11n_seg.py) for a YOLO segmentation scaffold with prototype masks
- [run_mask2former_numpy_postprocess.py](../examples/torch/run_mask2former_numpy_postprocess.py) for a more involved NumPy postprocess built from generic steps
