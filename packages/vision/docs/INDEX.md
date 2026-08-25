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

All operators in this section read and write named fields in a
`TensorRegistry`. By convention detection fields are `boxes`, `scores`, and
`classes`; segmentation registries can additionally carry `masks`.

| Operator | Input -> Output | Notes |
|---|---|---|
| `ConvertBoxFormat(...)` | `TensorRegistry` -> `TensorRegistry` | Converts between `xyxy`, `xywh`, and `cxcywh` box formats. |
| `NMS(...)` | `TensorRegistry` -> `TensorRegistry` | Applies confidence filtering and per-class non-maximum suppression. |
| `NMM(...)` | `TensorRegistry` -> `TensorRegistry` | Merges overlapping boxes instead of discarding them. |
| `FilterTensorsByScore(...)` | `TensorRegistry` -> `TensorRegistry` | Filters aligned fields by score threshold. |
| `FilterTensorsByClasses(...)` | `TensorRegistry` -> `TensorRegistry` | Filters aligned fields by allowed class ids. |
| `FilterTensorsByBoxArea(...)` | `TensorRegistry` -> `TensorRegistry` | Filters aligned fields by `xyxy` box area. |
| `FilterTensorsByMasksArea(...)` | `TensorRegistry` -> `TensorRegistry` | Filters aligned fields by foreground mask area. |
| `ProjectBoxes(src="boxes")` | `(TensorRegistry, ResizeTransform)` -> `TensorRegistry` | Projects model-space boxes into source-image coordinates. |
| `ReconstructMasks(...)` | `TensorRegistry` -> `TensorRegistry` | Reconstructs instance masks from coefficients and prototypes. |
| `ProjectMasks(...)` | `(TensorRegistry, ResizeTransform)` -> `TensorRegistry` | Projects prototype-style masks into source-image space. |
| `ProjectRoIMasks(...)` | `(TensorRegistry, ResizeTransform)` -> `TensorRegistry` | Projects per-instance RoI masks into source-image space. |
| `ResizeMasks(...)` | `(TensorRegistry, image_shape)` -> `TensorRegistry` | Resizes instance masks to an image shape. |
| `MasksToBoxes(...)` | `TensorRegistry` -> `TensorRegistry` | Derives boxes from binary masks. |
| `WeightMasksByScores(...)` | `TensorRegistry` -> `TensorRegistry` | Weights masks by per-instance scores. |
| `MeanMaskedScores(...)` | `TensorRegistry` -> `TensorRegistry` | Computes one score per instance from dense mask scores over foreground masks. |

## Tiling

| Operator | Input -> Output | Notes |
|---|---|---|
| `Tile(slice_wh, overlap_wh=(0, 0))` | `ImagePayload` -> `(list[ImagePayload], list[TileRect])` | Slices an image into overlapping crops for tiled inference. |
| `Stitch(*srcs, boxes="boxes")` | `(list[TensorRegistry], list[TileRect])` -> `TensorRegistry` | Remaps boxes and concatenates explicitly configured aligned tensors. |
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
