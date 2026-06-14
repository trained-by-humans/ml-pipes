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
`Normalize` is the single fixed-precision boundary: it converts uint8 input
to float, and its output dtype becomes the working precision for everything
that follows.

**Runtime-agnostic.** Operators use NumPy. They impose no dependency on
PyTorch, TensorFlow, or any specific hardware. `Infer` is the only step
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
| `Decode()` | `Path / str / bytes` → `ImagePayload` | Reads and decodes image file |
| `Resize(target_size, mode, interpolation, pad_value, center, allow_scale_up)` | `ImagePayload` → `(ImagePayload, ResizeTransform)` | `mode`: `"resize"` (stretch) or `"letterbox"` (aspect-ratio-preserving with padding) |
| `ConvertColorSpace(output_color_space)` | `ImagePayload` → `ImagePayload` | Converts between BGR and RGB while preserving layout/dtype |
| `Normalize(scale, mean, std, output_layout, output_color_space, add_batch_dim)` | `ImagePayload` → `TensorPayload` | Scales, normalizes, transposes layout, optionally converts BGR↔RGB |
| `Cast(dtype)` | `TensorPayload` → `TensorPayload` | Casts dtype, e.g. float32 → float16 |

### Inference

| Operator | Input → Output | Notes |
|---|---|---|
| `Infer(model_path, input_layout, dtype)` | `TensorPayload` → `RuntimeOutputs` | Runs ONNX Runtime. Validates layout and dtype contract before inference. |

### Registry creation

| Operator | Input → Output | Notes |
|---|---|---|
| `Extract(*names, as_=...)` | `RuntimeOutputs` → `TensorRegistry` | Extracts tensors by their ONNX graph output names. `as_` renames — pass a tuple for multi-output. |

### Output

| Operator | Input → Output | Notes |
|---|---|---|
| `ToDetections(boxes, scores, classes)` | `TensorRegistry` → `Detections` | Finalises a detection pipeline |
| `ToSegmentations(boxes, scores, classes, masks)` | `TensorRegistry` → `Segmentations` | Finalises a segmentation pipeline |
| `MapPredictionsToObjects(fields)` | `Detections / Segmentations` → `list[dict]` | Converts typed prediction arrays to a list of per-object dicts |

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
| `Squeeze(src, axis, as_)` | Removes unit dimensions. Without `axis`, removes all. |
| `Transpose(src, axes, as_)` | Permutes axes. Default reverses all axes. |

### Indexing

| Operator | Notes |
|---|---|
| `Slice(src, at, as_)` | Slices along the last axis: `Slice("preds", slice(None, 4), as_="boxes")` |
| `GatherRows(src, indices, as_)` | Indexes into a tensor along axis 0 |
| `FilterBy(src, indices, as_)` | Filters rows using an index array stored in the registry: `FilterBy("mask_coeffs", "kept")` |

### Math

| Operator | Notes |
|---|---|
| `ArgMax(src, axis, as_)` | Returns index of max value along axis (default -1) |
| `GatherScores(scores, classes, as_)` | Gathers `scores[i, classes[i]]` for each detection |
| `Softmax(src, axis, as_)` | Softmax along axis (default -1) |
| `Sigmoid(src, as_)` | Element-wise sigmoid |
| `Scale(src, by, as_)` | Multiplies by a scalar or per-column array. Use `by=(W, H, W, H)` to denormalize `cxcywh` boxes, `by=1/255` to normalize pixel values. |

### Geometry

| Operator | Notes |
|---|---|
| `ConvertBoxFormat(src, *, from_, to, as_)` | Converts between `"xyxy"`, `"xywh"`, `"cxcywh"`. `src` defaults to `"boxes"`, `to` defaults to `"xyxy"`. |

### Detection

| Operator | Notes |
|---|---|
| `NMS(boxes, scores, classes, conf_threshold, iou_threshold, max_detections, kept_as)` | Non-maximum suppression. Set `kept_as` to store kept indices for downstream `FilterBy` calls. |
| `NMM(iou_threshold)` | Non-maximum merge. Groups overlapping detections per class and replaces each group with a single score-weighted average box. Unlike `NMS`, no detection is discarded — overlapping boxes are merged into one. |

### Segmentation

| Operator | Notes |
|---|---|
| `ReconstructMasks(coefficients, prototypes, as_)` | Matrix multiply: `(N, C) @ (C, H*W)` → `(N, H, W)`. `as_` is required — no default — to keep the output name explicit. |

### Projection

These operators accept `(TensorRegistry, ResizeTransform)`. Use `Recall` before
them to inject the stored transform.

| Operator | Notes |
|---|---|
| `ProjectBoxes(src)` | Inverse-transforms boxes from model input space to original image space, accounting for scale and letterbox padding. |
| `ProjectMasks(masks, boxes, mask_threshold)` | Zeros prototype masks outside each bounding box (in prototype space, vectorised across all N masks), then upsamples to original image size. Boxes are converted from original image space to prototype space internally. **Must be called after `ProjectBoxes`**. |
| `ProjectRoIMasks(masks, boxes, mask_threshold)` | Resizes per-instance RoI masks `(N, H, W)` to their bounding boxes and embeds them into a full-image canvas. **Must be called after `ProjectBoxes`** — needs boxes in original image space. For `(N, 1, H, W)` outputs, add `Squeeze("masks", axis=1)` first. |

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
| `Select(*selector)` | Projects the current value via attribute access and tuple indexing. Example: `Select("spatial_shape", 0)`, `Select("spatial_shape.0")`, or `Select("array")`. |
| `Pick(index)` | Tuple-only shorthand for selecting one element from a tuple and discarding the rest. |

---

## Parallelism operators

Parallelism operators let a single pipeline call fan out work across multiple
threads and collect the results. They come in matched pairs: `Scatter` marks
the start of a parallel region; `Gather` marks the end.

The input to `Scatter` must be a `list`. Each item is dispatched to a worker
thread that runs the enclosed region independently with a fresh `Context`.
The original thread blocks at `Gather` until all workers finish, then resumes
with `list[results]` in submission order.

```
                 ┌─ worker 0: [region ops] ─┐
list[T] ─ Scatter┼─ worker 1: [region ops] ─┼─ Gather ─ list[U]
                 └─ worker 2: [region ops] ─┘
```

| Operator | Notes |
|---|---|
| `Scatter(max_concurrency)` | Fans `list[T]` out to worker threads. `max_concurrency` bounds the thread pool size; defaults to `1` (sequential). |
| `Gather()` | Collects worker results back into `list[U]`. Must follow a matching `Scatter`. |

**Constraints:**
- Scatter/Gather cannot be nested inside another Scatter region.
- A Batch/UnBatch region inside a Scatter region is valid.
- If any worker raises, the exception propagates on the original thread after all workers complete.

**Example — tiled inference:**

```python
pipeline = Pipeline([
    Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
    Store("tile_rects", index=1),
    Pick(0),
    Scatter(max_concurrency=4),
    Decode(),
    Resize((640, 640)),
    Normalize(),
    Infer("model.onnx"),
    ...
    ToDetections(),
    Gather(),
    Recall("tile_rects"),
    Stitch(),
    NMM(iou_threshold=0.5),
])
```

---

## Tiling operators

Tiling operators split an image into overlapping crops for inference and
reassemble the per-tile detections back into the original image coordinate
space. They are designed to work together with `Scatter`/`Gather` to
run inference on each tile in parallel.

| Operator | Notes |
|---|---|
| `Tile(slice_wh, overlap_wh)` | Splits `ImagePayload` into overlapping crops. Returns `(list[ImagePayload], list[TileRect])`. `slice_wh` is `(width, height)` of each tile; `overlap_wh` is the overlap in pixels (default `(0, 0)`). |
| `Stitch()` | Remaps each tile's `Detections` boxes from tile coordinates to original image coordinates and concatenates all tiles. Returns `Detections`. Accepts `(list[Detections], list[TileRect])`. |
| `TileRect` | Frozen dataclass `(x1, y1, x2, y2)` describing a crop window in the original image. Produced by `Tile` and consumed by `Stitch`. |

`Stitch` performs pure coordinate remapping and concatenation — it does not
deduplicate cross-tile detections. Follow it with `NMM` (or `NMS`) to merge
overlapping boxes that span tile boundaries.

---

## Side-effect operators

Side-effect operators tap the pipeline for logging, drawing, or saving. They
pass the input value through unchanged.

| Operator | Notes |
|---|---|
| `DrawBoxes()` | Draws bounding boxes on an `ImagePayload` |
| `SaveImage(path)` | Saves an `ImagePayload` to disk |
| `LogDetections(model_path, image_path, annotated_image_path)` | Logs detection results as JSON to stdout |
