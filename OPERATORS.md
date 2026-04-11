# Operators

Operators are the building blocks of a pipeline. They fall into four families:
[Transform](#transform-operators) · [Tensor](#tensor-operators) · [Context](#context-operators) · [Side-effect](#side-effect-operators)

---

## Design Principles

Every operator in the library is designed to uphold the following properties.
They are not style guidelines — they are what makes operators safe to compose
and swap without side effects.

**Stateless.** An operator holds only the configuration given at construction
time (`name`, `axis`, `threshold`, etc.). It has no mutable state that
accumulates between calls. Calling it twice on the same input produces the
same output.

**Single-responsibility.** Each operator does exactly one thing. `Squeeze`
removes unit dimensions. `ConvertBoxFormat` converts between coordinate
formats. `NMS` filters by confidence and IoU. Complexity is built by
composing simple operators, not by adding parameters to existing ones.

**Model-agnostic.** No operator knows which model produced the tensors it
processes. `NMS`, `ProjectBoxes`, and `Softmax` are generic. Model-specific
adaptations live in the pipeline list as individual operators, not inside
shared infrastructure.

**Precision-agnostic.** Operators preserve the dtype of their input. A
pipeline that runs in float32 runs in float16 without modifying any operator.
`NormalizeOp` is the single fixed-precision boundary: it converts uint8 input
to float, and its output dtype becomes the working precision for everything
that follows.

**Runtime-agnostic.** Operators use NumPy. They impose no dependency on
PyTorch, TensorFlow, or any specific hardware. `InferOp` is the only step
that touches a runtime; everything before and after is plain NumPy and
transfers to any compute environment.

**Composable.** Every operator has the same contract: receive a value, return
a value. Any Python callable fits. Pipelines are plain lists — operators can
be reordered, replaced, or inserted without touching anything else.

## Transform operators

Transform operators convert data from one type to another. The type of the
flowing value changes at each step.

### Preprocessing

Image file into an inference-ready tensor:

| Operator | Input → Output | Notes |
|---|---|---|
| `DecodeOp()` | `Path / str / bytes` → `ImagePayload` | Reads and decodes image file |
| `ResizeOp(target_size, mode, interpolation, pad_value, center, allow_scale_up)` | `ImagePayload` → `(ImagePayload, ResizeTransform)` | `mode`: `"resize"` (stretch) or `"letterbox"` (aspect-ratio-preserving with padding) |
| `NormalizeOp(scale, mean, std, output_layout, output_color_space, add_batch_dim)` | `ImagePayload` → `TensorPayload` | Scales, normalizes, transposes layout, optionally converts BGR↔RGB |
| `CastTensorOp(dtype)` | `TensorPayload` → `TensorPayload` | Casts dtype, e.g. float32 → float16 |

### Inference

| Operator | Input → Output | Notes |
|---|---|---|
| `InferOp(model_path, expected_input_layout, expected_model_dtype)` | `TensorPayload` → `RuntimeOutputs` | Runs ONNX Runtime. Validates layout and dtype contract before inference. |

### Registry creation

| Operator | Input → Output | Notes |
|---|---|---|
| `Select(*names, as_=...)` | `RuntimeOutputs` → `TensorRegistry` | Extracts tensors by their ONNX graph output names. `as_` renames — pass a tuple for multi-output. |

### Output

| Operator | Input → Output | Notes |
|---|---|---|
| `ToDetections(boxes, scores, classes)` | `TensorRegistry` → `Detections` | Finalises a detection pipeline |
| `ToSegmentations(boxes, scores, classes, masks)` | `TensorRegistry` → `Segmentations` | Finalises a segmentation pipeline |
| `MapToObjectsOp(field_sources)` | `Detections / Segmentations` → `list[dict]` | Converts typed prediction arrays to a list of per-object dicts |

---

## Tensor operators

Tensor operators receive a `TensorRegistry` and return a `TensorRegistry`.
The type does not change — only the values inside the registry are transformed.
All tensor operators accept an optional `as_` parameter: omit it to overwrite
the source tensor in-place; provide it to write to a new key.

```python
Squeeze("preds")                                    # overwrites "preds"
Slice("preds", slice(None, 4), as_="boxes")         # creates "boxes", "preds" unchanged
```

### Shape

| Operator | Notes |
|---|---|
| `Squeeze(name, axis, as_)` | Removes unit dimensions. Without `axis`, removes all. |
| `Transpose(name, axes, as_)` | Permutes axes. Default reverses all axes. |

### Indexing

| Operator | Notes |
|---|---|
| `Slice(name, slice, as_)` | Slices along the last axis: `Slice("preds", slice(None, 4), as_="boxes")` |
| `Gather(name, indices, as_)` | Indexes into a tensor along axis 0 |
| `FilterBy(name, indices, as_)` | Filters rows using an index array stored in the registry: `FilterBy("mask_coeffs", "kept")` |

### Math

| Operator | Notes |
|---|---|
| `ArgMax(name, axis, as_)` | Returns index of max value along axis (default -1) |
| `GatherScores(scores, classes, as_)` | Gathers `scores[i, classes[i]]` for each detection |
| `Softmax(name, axis, as_)` | Softmax along axis (default -1) |
| `Sigmoid(name, as_)` | Element-wise sigmoid |
| `Scale(name, by, as_)` | Multiplies by a scalar or per-column array. Use `by=(W, H, W, H)` to denormalize `cxcywh` boxes, `by=1/255` to normalize pixel values. |

### Geometry

| Operator | Notes |
|---|---|
| `ConvertBoxFormat(name, from_, to, as_)` | Converts between `"xyxy"`, `"xywh"`, `"cxcywh"` |

### Detection

| Operator | Notes |
|---|---|
| `NMS(boxes, scores, classes, conf_threshold, iou_threshold, max_detections, kept_as)` | Non-maximum suppression. Set `kept_as` to store kept indices for downstream `FilterBy` calls. |

### Segmentation

| Operator | Notes |
|---|---|
| `ReconstructMasks(coefficients, prototypes, dst)` | Matrix multiply: `(N, C) @ (C, H*W)` → `(N, H, W)`. `dst` is required — no default — to keep the output name explicit. |

### Projection

These operators accept `(TensorRegistry, ResizeTransform)`. Use `Recall` before
them to inject the stored transform.

| Operator | Notes |
|---|---|
| `ProjectBoxes(name)` | Inverse-transforms boxes from model input space to original image space, accounting for scale and letterbox padding. |
| `ProjectMasks(masks, boxes, mask_threshold)` | Crops prototype masks at box coordinates, upsamples to original image size, and thresholds. **Must be called before `ProjectBoxes`** — needs boxes still in model space. |
| `ProjectRoIMasks(masks, boxes, mask_threshold)` | Resizes per-instance RoI masks (e.g. Mask R-CNN's 28×28 outputs) to their bounding boxes and embeds them into a full-image canvas. Accepts `(N, H, W)` or `(N, 1, H, W)`. **Must be called after `ProjectBoxes`** — needs boxes in original image space. |

---

## Context operators

Context operators manage the side-channel that lets values computed early in
the pipeline (e.g. the resize transform) be accessed later without threading
them through every operator in between. See the
[Context section in the README](README.md#context) for a full explanation.

| Operator | Notes |
|---|---|
| `Store(name, index)` | Saves the current value (or `current[index]`) into context. The flowing value is unchanged. |
| `Recall(name)` | Appends a stored value to the flowing value, producing a tuple. Idempotent. |
| `Pick(index)` | Selects one element from a tuple, discarding the others. |

---

## Side-effect operators

Side-effect operators tap the pipeline for logging, drawing, or saving. They
pass the input value through unchanged.

| Operator | Notes |
|---|---|
| `DrawBoxesOp()` | Draws bounding boxes on an `ImagePayload` |
| `SaveImageOp(path)` | Saves an `ImagePayload` to disk |
| `LogDetectionsOp(model_path, image_path, annotated_image_path)` | Logs detection results as JSON to stdout |
