# Vision Task And Media Domain

`ml_pipes.vision` owns image inputs, reusable vision semantics, and typed
vision outputs inside `ml-pipes`.

For the full package surface and operator catalog, see [`INDEX.md`](./INDEX.md).

## Package Profile

| Dimension       | Classification                      |
|-----------------|-------------------------------------|
| Role / Function | `Data prep`, `Inference (Scaffold)` |
| Task Type       | `Vision`                            |
| Data Type       | `Media`, `Tensors`                  |

## Scope And Use Cases

`ml_pipes.vision` focuses on the vision-specific stages around
inference: preprocessing before the runtime step, and task-specific
postprocess after the Tensor domain. It carries the value types and operators
that keep image semantics, resize/projection metadata, tiling, and typed
vision results explicit.

Use this package when a pipeline starts from image inputs, needs
vision-specific preprocessing before inference, or needs final tensors turned
into typed vision results such as detections, segmentations, or density
predictions.

> [!NOTE]
> Vision does not own runtime execution or generic tensor-space operators.
> Use runtime packages such as `ml_pipes.onnx` or `ml_pipes.torch` for
> inference and `ml_pipes.tensor` for shared tensor postprocess.

## Design Principles

- Keep image and resize metadata explicit with `ImagePayload` and
  `ResizeTransform`.
- Keep image and task semantics here, and hand off shared tensor math or
  runtime execution to their owning packages.
- Return typed predictions instead of anonymous dicts so downstream filtering,
  logging, and rendering stay coherent.
- Make side-effect operators opt-in and pass-through so pipelines can stay
  explicit about where files or logs are produced.

## Intermediate And Terminal Values

Vision values usually fall into two groups: intermediate values that carry
image or geometry state through the pipeline, and terminal values that
represent finalized task results.

### Intermediate Values

`ImagePayload`, `ResizeTransform`, and `TileRect` are the main intermediate
values in this package.

`ImagePayload` keeps image data together with color-space and layout metadata
during preprocessing and rendering. `ResizeTransform` records geometry changes
from steps such as `Resize()` so later operators such as `ProjectBoxes()` or
`ProjectMasks()` can map model-space results back to the source image space.
`TileRect` records where each tile came from so `Stitch()` can merge
tile-local results back into one full-image result.

### Terminal Values

`Detections`, `Segmentations`, and `DensityPrediction` are the terminal
task-result values in this package.

These values mark the point where tensor-space results become typed vision
results that later steps can filter, render, log, or map into user-facing
objects.

## Where Vision Fits

Vision usually appears at both ends of an inference pipeline: it prepares
image inputs before runtime, then finalizes task results after generic tensor
postprocess.

At a high level, a common flow looks like this:

```text
┌──────────────────────────────────────────────────────────┐
│ Vision Input                                             │
├─ LoadFile -> Decode -> Resize -> Normalize               │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorPayload
         ▼
┌──────────────────────────────────────────────────────────┐
│ Runtime + Tensor Domains                                 │
├─ Infer / TorchInfer -> Extract -> Slice -> ArgMax -> ... │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorRegistry
         ▼
┌──────────────────────────────────────────────────────────┐
│ Vision Tensor Postprocess                                │
├─ ProjectBoxes / ProjectMasks / ResizeMasks -> ...        │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorRegistry
         ▼
┌──────────────────────────────────────────────────────────┐
│ Typed Prediction Conversion                              │
├─ ToDetections / ToSegmentations / ToDensityPrediction    │
└────────┬─────────────────────────────────────────────────┘
         |
         | typed predictions
         ▼
┌──────────────────────────────────────────────────────────┐
│ Typed Prediction Processing                              │
├─ FilterPredictions / DrawBoxes / SaveImage / ...         │
└──────────────────────────────────────────────────────────┘
```

That split keeps image preparation, vision-specific tensor postprocess, typed
prediction conversion, and prediction-side processing as separate reusable
stages.

## Using Vision In Pipelines

This section focuses on the Vision-owned stages before runtime and after
shared tensor postprocess.

### Prepare Images For Runtime

Use the Vision package up front when the pipeline starts from files, bytes, or
raw image arrays and needs explicit image preparation before inference.

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Store
from ml_pipes.vision import Decode, LoadFile, Normalize, Resize

pipeline = Pipeline([
    LoadFile(),
    Decode(),
    Resize((640, 640)),
    Store("resize_transform", source=1),
    Pick(0),
    Normalize(),
])
```

This is the front half of many example pipelines, including
[`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py).

### Vision-Specific Postprocess

After runtime and shared tensor postprocess, Vision handles the remaining
vision-specific work: project boxes or masks, reconstruct or resize masks when
needed, apply vision-specific filtering, and convert the registry values into
typed predictions.

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall
from ml_pipes.vision import ConvertBoxFormat, NMS, ProjectBoxes, ToDetections

pipeline = Pipeline([
    ...,
    ConvertBoxFormat(from_="cxcywh"),
    NMS(),
    Recall("resize_transform"),
    ProjectBoxes(),
    ToDetections(),
])
```

For segmentation pipelines, the same package owns mask reconstruction,
projection, and final `ToSegmentations(...)` handoff. See
[`examples/run_yolo11n_seg.py`](../../../examples/run_yolo11n_seg.py).

### Prediction-Based Operators

Once Vision has converted registry values into typed predictions, later Vision
operators can work on those typed results directly. This is where
prediction-side filtering, rendering, logging, mapping, or saving usually
happens.

For example, after `ToDetections()` the pipeline can recall the source image,
draw the detections, and save the rendered image:

```python
from pathlib import Path

from ml_pipes.core import Pipeline
from ml_pipes.standard import Recall
from ml_pipes.vision import DrawBoxes, SaveImage, ToDetections

pipeline = Pipeline([
    ...,
    ToDetections(),
    Recall("source_image", prepend=True),
    DrawBoxes(),
    SaveImage(Path("result.jpg"), at=0),
])
```

### Tile Large Images And Stitch Results Back

Use the Vision package when inference must happen over tiles rather than one
full-frame image.

```python
from ml_pipes.core import Inline, Pipeline
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Store
from ml_pipes.vision import NMM, Stitch, Tile

pipeline = Pipeline([
    Tile(slice_wh=(320, 320), overlap_wh=(80, 80)),
    Store("tile_rects", source=1),
    Pick(0),
    Scatter(max_concurrency=4),
    Inline(tile_pipeline),
    Gather(),
    Recall("tile_rects"),
    Stitch(),
    NMM(iou_threshold=0.4),
])
```

This shape appears in
[`examples/run_yolo8_tile.py`](../../../examples/run_yolo8_tile.py).

## Further Reading

- [`INDEX.md`](./INDEX.md) for the full surface catalog
- [`docs/README.md`](../../../docs/README.md) for the shared framework docs index
- [`Tensor guide`](../../tensor/docs/README.md) for the shared tensor domain
  that Vision builds on
- [`examples/README.md`](../../../examples/README.md) for runnable pipeline entry points
  - [`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py) for a
    baseline end-to-end detection pipeline
  - [`examples/run_yolo8_tile.py`](../../../examples/run_yolo8_tile.py) for
    tiled detection over large images
  - [`examples/streaming/run_shibuya_csrnet.py`](../../../examples/streaming/run_shibuya_csrnet.py)
    for density prediction and overlay rendering
