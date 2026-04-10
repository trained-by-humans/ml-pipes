# ml-pipes

A composable ONNX inference pipeline library. Pipelines are built by chaining
generic operators — no subclassing, no model-specific base classes. Each
operator does one thing; different models and tasks are handled by composing
the right operators in the right order.

## Install

```bash
pip install -e .
```

## Core concepts

### Pipeline

A `Pipeline` is a list of callables executed in sequence. Each step receives
the output of the previous step as its input.

```python
from ml_pipes import Pipeline, DecodeOp, ResizeOp, NormalizeOp, InferOp

pipeline = Pipeline([
    DecodeOp(),
    ResizeOp((640, 640)),
    NormalizeOp(),
    InferOp("model.onnx"),
    ...
])

result = pipeline("image.jpg")
```

Any plain callable (function or lambda) can be dropped into the pipeline
alongside operators — useful for model-specific logic that doesn't warrant a
dedicated operator.

### TensorRegistry

After inference, outputs are extracted into a `TensorRegistry` — a mutable
named store of NumPy arrays. All post-processing operators read from and write
to the registry by tensor name.

```python
Select("output0", as_="preds")   # RuntimeOutputs → TensorRegistry
Squeeze("preds")                  # (1, 116, N) → (116, N), in-place
Transpose("preds")                # (116, N) → (N, 116), in-place
```

All tensor operators accept an optional `as_` parameter. When omitted the
result overwrites the source tensor (in-place). When provided a new tensor is
created under that name.

### Context — Store / Recall

`Store` and `Recall` thread values across pipeline phases via an immutable
side-channel. This is how a resize transform computed during preprocessing is
made available during postprocessing without passing it through every operator
in between.

```python
ResizeOp((640, 640)),
Store("resize_transform", index=1),   # store the transform, keep image flowing
Pick(0),                               # drop the transform from current value
...
Recall("resize_transform"),            # append transform → (registry, transform)
ProjectBoxes(),                        # receives (registry, transform)
```

---

## Operators

### Pre-processing

| Operator | Description |
|---|---|
| `DecodeOp()` | Decode an image file path or bytes into an `ImagePayload` |
| `ResizeOp(target_size, mode, ...)` | Resize or letterbox an image |
| `NormalizeOp(scale, mean, std, output_layout, output_color_space, add_batch_dim)` | Scale, normalize, transpose layout, convert color space |
| `CastTensorOp(dtype)` | Cast tensor dtype (e.g. float32 → float16) |

### Inference

| Operator | Description |
|---|---|
| `InferOp(model_path, expected_input_layout, expected_model_dtype)` | Run ONNX Runtime inference, returns `RuntimeOutputs` |

### Registry

| Operator | Description |
|---|---|
| `Select(*names, as_=...)` | Extract named tensors from `RuntimeOutputs` into a `TensorRegistry` |

### Shape

| Operator | Description |
|---|---|
| `Squeeze(name, axis, as_)` | Remove unit dimensions |
| `Transpose(name, axes, as_)` | Permute axes |

### Indexing

| Operator | Description |
|---|---|
| `Slice(name, slice, as_)` | Slice a tensor along the last axis |
| `Gather(name, indices, as_)` | Index into a tensor |
| `FilterBy(name, indices, as_)` | Filter tensor rows by an index array stored in the registry |

### Math

| Operator | Description |
|---|---|
| `ArgMax(name, axis, as_)` | Argmax along an axis |
| `GatherScores(scores, classes, as_)` | Gather per-detection scores at class indices |
| `Softmax(name, axis, as_)` | Softmax along an axis |
| `Sigmoid(name, as_)` | Element-wise sigmoid |
| `Scale(name, by, as_)` | Multiply by a scalar or per-column array — use for normalizing or denormalizing coordinates |

### Geometry

| Operator | Description |
|---|---|
| `ConvertBoxFormat(name, from_, to, as_)` | Convert between `xyxy`, `xywh`, `cxcywh` |

### Detection

| Operator | Description |
|---|---|
| `NMS(boxes, scores, classes, conf_threshold, iou_threshold, max_detections, kept_as)` | Non-maximum suppression. Set `kept_as` to store the kept indices for use with `FilterBy`. |

### Segmentation

| Operator | Description |
|---|---|
| `ReconstructMasks(coefficients, prototypes, dst)` | Reconstruct masks via `coeffs @ protos` (YOLO-style prototype-based segmentation) |

### Projection

| Operator | Description |
|---|---|
| `ProjectBoxes(name)` | Inverse-transform boxes from model input space to original image space. Requires `Recall` of a `ResizeTransform`. |
| `ProjectMasks(masks, boxes, mask_threshold)` | Crop, upsample, and threshold prototype masks. Must be called **before** `ProjectBoxes`. Requires `Recall` of a `ResizeTransform`. |

### Output

| Operator | Description |
|---|---|
| `ToDetections(boxes, scores, classes)` | Convert `TensorRegistry` to a `Detections` dataclass |
| `ToSegmentations(boxes, scores, classes, masks)` | Convert `TensorRegistry` to a `Segmentations` dataclass |

### Context

| Operator | Description |
|---|---|
| `Store(name, index)` | Store a value (or element of a tuple) into the pipeline context |
| `Recall(name)` | Append a stored value to the current pipeline value |
| `Pick(index)` | Select one element from a tuple |

### Side effects

| Operator | Description |
|---|---|
| `DrawBoxesOp()` | Draw bounding boxes on an image |
| `SaveImageOp(path)` | Save an image to disk |
| `MapToObjectsOp(field_sources)` | Map detection fields to a list of structured objects |
| `LogDetectionsOp(model_path, image_path, annotated_image_path)` | Log detections to stdout as JSON |

---

## Examples

All examples download the model and a COCO validation image on first run into
`.example_assets/` and write an annotated output image to the same directory.

### Generic detection (bring your own model)

```bash
python examples/run_detection.py path/to/model.onnx path/to/image.jpg
```

### YOLOv8n — object detection

```bash
python examples/run_yolo8n_onnx.py
```

### YOLO11n FP16 — object detection (letterbox + float16)

```bash
python examples/run_yolo11n_onnx_fp16.py
```

Demonstrates `CastTensorOp` for FP16 inference and letterbox resize with
center padding.

### RF-DETR nano — transformer-based detection

```bash
python examples/run_rfdetr_nano_onnx.py
```

RF-DETR outputs normalized `(cx, cy, w, h)` boxes. The pipeline uses `Scale`
to convert to pixel coordinates before box format conversion and NMS.

### YOLO11n-seg — instance segmentation (prototype-based)

```bash
python examples/run_yolo11n_seg_onnx.py
```

Demonstrates the prototype-based mask pipeline:
`ReconstructMasks` → `ProjectMasks` → `ProjectBoxes`.

### Mask R-CNN int8 — instance segmentation (CNN family)

```bash
python examples/run_maskrcnn_onnx.py
```

Structurally different from the YOLO segmentation pipeline: NMS is baked into
the model, masks are per-instance 28×28 RoI probabilities rather than
prototype coefficients, and preprocessing uses BGR mean subtraction without
normalizing to [0, 1].

---

## Pipeline contracts

Operators declare their input and output types via Python type annotations.
`Pipeline(validate_on_init=True)` checks that each operator's input type is
compatible with the previous operator's output type at construction time,
catching mismatches before any data flows through.

```python
pipeline = Pipeline([...], validate_on_init=True)
```
