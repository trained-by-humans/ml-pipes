# Operator reference

Operators are the building blocks of a pipeline. They fall into four families:
[Transform](#transform-operators) · [Tensor](#tensor-operators) · [Context](#context-operators) · [Side-effect](#side-effect-operators)

---

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
