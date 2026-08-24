# ml-pipes-vision Index

This page catalogs the vision package surface in `ml_pipes.vision`.
For package overview, scope, design principles, and usage patterns, see
[`README.md`](./README.md).

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md).
For the cross-package package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Package Primitives

| Surface | Notes |
|---|---|
| `ImagePayload` | Main image boundary type. |
| `ResizeTransform` | Carries resize metadata needed later for projection back to source-image space. |
| `TileRect` | Describes one tile in original-image coordinates. |

## Input And Preprocessing

| Operator | Input -> Output | Notes |
|---|---|---|
| `LoadFile()` | `Path / str` -> `bytes` | Reads a file into raw bytes. |
| `Decode()` | `bytes` -> `ImagePayload` | Decodes image bytes into the vision payload model. |
| `Resize(target_size, mode, ...)` | `ImagePayload` -> `(ImagePayload, ResizeTransform)` | Resizes or letterboxes an image and returns the resize metadata needed later for projection. |
| `ConvertColorSpace(output_color_space)` | `ImagePayload` -> `ImagePayload` | Converts RGB and BGR images while preserving layout. |
| `Normalize(...)` | `ImagePayload` -> `TensorPayload` | Scales, normalizes, reorders layout, and optionally adds a batch dimension. |

## Detection And Segmentation Registry Helpers

| Operator | Notes |
|---|---|
| `ConvertBoxFormat(src="boxes", from_=..., to="xyxy", as_=None)` | Converts between `xyxy`, `xywh`, and `cxcywh` box formats. |
| `NMS(...)` | Confidence filtering plus per-class non-maximum suppression on registry tensors. |
| `NMM(boxes="boxes", scores="scores", classes="classes", ...)` | Merges overlapping registry detections instead of discarding them. |
| `FilterTensorsByBoxArea(...)` | Filters one or more tensors by `xyxy` box area. |
| `FilterTensorsByScore(...)` | Filters one or more tensors by a score threshold. |
| `FilterTensorsByClasses(...)` | Filters one or more tensors by allowed class ids. |
| `FilterTensorsByMasksArea(...)` | Filters one or more tensors by mask area. |
| `WeightMasksByScores(...)` | Weights masks by per-instance scores. |
| `ResizeMasks(...)` | Resizes instance masks to an image shape. |
| `MeanMaskScores(...)` | Computes mean scores over masks or masked areas. |
| `MasksToBoxes(...)` | Derives boxes from binary masks. |
| `ReconstructMasks(coefficients, prototypes, as_)` | Reconstructs instance masks from coefficients and prototypes. |
| `ProjectBoxes(src="boxes")` | Projects model-space boxes back to the original image space. |
| `ProjectMasks(...)` | Projects prototype-style masks back to the source image space. |
| `ProjectRoIMasks(...)` | Projects per-instance RoI masks back to the source image space. |

## Tiling

| Operator | Input -> Output | Notes |
|---|---|---|
| `Tile(slice_wh, overlap_wh=(0, 0))` | `ImagePayload` -> `(list[ImagePayload], list[TileRect])` | Slices an image into overlapping crops for tiled inference. |
| `Stitch(...)` | `(list[TensorRegistry], list[TileRect])` -> `TensorRegistry` | Remaps tile-local detection tensors back to the original image space. |
| `TileRect` | value type | Describes one tile in original-image coordinates. |

## Rendering And Side Effects

| Operator | Notes |
|---|---|
| `DrawBoxes(...)` | Draws configured detection tensors on an image while passing the registry through. |
| `DrawMasks(...)` | Overlays configured mask tensors while passing the registry through. |
| `SaveImage(output_path, at=None)` | Saves an image payload to disk as a side effect. |
| `LogDetections(...)` | Logs configured detection tensors as JSON as a side effect. |

## Density

| Operator | Input -> Output | Notes |
|---|---|---|
| `ClampDensity(src="density", as_=None)` | `TensorRegistry` -> `TensorRegistry` | Clamps a density tensor to non-negative values. |
| `SumDensity(src="density")` | `TensorRegistry` -> `float` | Sums a density tensor into one count. |
| `DensityToHeatmap(src="density", ...)` | `(ImagePayload, TensorRegistry)` -> `(ImagePayload, ImagePayload)` | Converts a density tensor into a colored heatmap aligned to the source image. |
| `BlendImages(...)` | `(ImagePayload, ImagePayload)` -> `ImagePayload` | Blends a source image with an overlay image. |
