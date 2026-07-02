# ml-pipes-vision Operator Index

This file catalogs the vision operators shipped with `ml-pipes-vision`
through `ml_pipes.vision`.

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md). For the cross-package
package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

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
| `NMM(iou_threshold=0.5)` | Merges overlapping detections instead of discarding them. |
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

## Prediction Outputs And Filtering

| Operator | Input -> Output | Notes |
|---|---|---|
| `ToDetections(...)` | `TensorRegistry` -> `Detections` | Finalizes a detection result. |
| `ToSegmentations(...)` | `TensorRegistry` -> `Segmentations` | Finalizes a segmentation result. |
| `FilterPredictions(predicate)` | prediction -> prediction | Filters typed prediction objects with a custom predicate. |
| `FilterPredictionsByClass(classes)` | prediction -> prediction | Filters typed prediction objects by class id. |
| `FilterPredictionsByScore(min_score)` | prediction -> prediction | Filters typed prediction objects by score. |
| `FilterPredictionsByArea(...)` | prediction -> prediction | Filters typed prediction objects by box area. |
| `MapPredictionsToObjects(fields, at=None)` | prediction -> `list[dict]` | Converts typed predictions into per-object mappings. |

## Tiling

| Operator | Input -> Output | Notes |
|---|---|---|
| `Tile(slice_wh, overlap_wh=(0, 0))` | `ImagePayload` -> `(list[ImagePayload], list[TileRect])` | Slices an image into overlapping crops for tiled inference. |
| `Stitch()` | `(list[Detections], list[TileRect])` -> `Detections` | Remaps tile-local detections back to the original image space. |
| `TileRect` | value type | Describes one tile in original-image coordinates. |

## Rendering And Side Effects

| Operator | Notes |
|---|---|
| `DrawBoxes(...)` | Draws detection boxes on an image while passing detections through. |
| `DrawMasks(...)` | Overlays segmentation masks on an image while passing segmentations through. |
| `SaveImage(output_path, at=None)` | Saves an image payload to disk as a side effect. |
| `LogDetections(...)` | Logs detection objects as JSON as a side effect. |

## Density

| Operator | Input -> Output | Notes |
|---|---|---|
| `ToDensityPrediction(src="density")` | `TensorRegistry` -> `DensityPrediction` | Finalizes a density-map result. |
| `ClampDensity()` | `DensityPrediction` -> `DensityPrediction` | Clamps density values to non-negative values. |
| `SumDensity()` | `DensityPrediction` -> `float` | Sums the density map into one count. |
| `DensityToHeatmap(...)` | `(ImagePayload, DensityPrediction)` -> `(ImagePayload, ImagePayload)` | Converts a density map into a colored heatmap aligned to the source image. |
| `BlendImages(...)` | `(ImagePayload, ImagePayload)` -> `ImagePayload` | Blends a source image with an overlay image. |
