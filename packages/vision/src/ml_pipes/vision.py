from __future__ import annotations

import json
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Literal, TextIO, TypeAlias, TypeVar, cast, get_args, get_origin, overload

import numpy as np

from ._typing.annotation import is_assignable, is_concrete_annotation
from .density import BlendImages, ClampDensity, DensityPrediction, DensityToHeatmap, SumDensity, ToDensityPrediction
from .operator import Operator
from .standard import SideEffectOp
from .tensor import FilterTensors, _flatten_leading_dim
from .tensor_types import TensorRegistry, TensorPayload
from .tiling import Stitch, Tile, TileRect
from .validation import PipelineValidationError
from .vision_types import (
    BoxPrediction,
    ClassPrediction,
    Detections,
    ImagePayload,
    Prediction,
    PredictionMask,
    ResizeTransform,
    ScorePrediction,
    Segmentations,
)

__all__ = [
    "BlendImages",
    "ClampDensity",
    "ConvertBoxFormat",
    "ConvertColorSpace",
    "Decode",
    "DensityPrediction",
    "DensityToHeatmap",
    "Detections",
    "DrawBoxes",
    "DrawMasks",
    "FilterPredictions",
    "FilterPredictionsByArea",
    "FilterPredictionsByClass",
    "FilterPredictionsByScore",
    "FilterTensorsByClasses",
    "FilterTensorsByMasksArea",
    "FilterTensorsByScore",
    "ImagePayload",
    "LoadFile",
    "LogDetections",
    "MapPredictionsToObjects",
    "MasksToBoxes",
    "MeanMaskScores",
    "NMM",
    "NMS",
    "Normalize",
    "Prediction",
    "ProjectBoxes",
    "ProjectMasks",
    "ProjectRoIMasks",
    "ReconstructMasks",
    "Resize",
    "ResizeMasks",
    "ResizeTransform",
    "SaveImage",
    "Segmentations",
    "Stitch",
    "SumDensity",
    "Tile",
    "TileRect",
    "ToDensityPrediction",
    "ToDetections",
    "ToSegmentations",
    "WeightMasksByScores",
]

DetectT = TypeVar("DetectT", bound=Detections)
SegT = TypeVar("SegT", bound=Segmentations)
PredictionT = TypeVar("PredictionT", bound=Prediction)
ClassPredictionT = TypeVar("ClassPredictionT", bound=ClassPrediction)
ScorePredictionT = TypeVar("ScorePredictionT", bound=ScorePrediction)
BoxPredictionT = TypeVar("BoxPredictionT", bound=BoxPrediction)
PayloadT = TypeVar("PayloadT")
ObjectPrefixT = TypeVar("ObjectPrefixT")
ObjectIndexT = TypeVar("ObjectIndexT", bound=int | None)
ObjectMapping: TypeAlias = dict[str, object]


def _is_unresolved_object_list(annotation: Any) -> bool:
    return annotation is list or (get_origin(annotation) is list and not is_concrete_annotation(annotation))


@Operator
class LoadFile:
    def __call__(self, image_path: str | Path) -> bytes:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()


@Operator
class Decode:
    def __call__(self, data: bytes) -> ImagePayload:
        import cv2

        image_bytes = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return ImagePayload(array=image, color_space="BGR", layout="HWC")


@Operator
class Resize:
    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        mode: Literal["letterbox", "resize"] = "letterbox",
        pad_value: int = 114,
        interpolation: Literal["nearest", "linear", "cubic", "area"] = "linear",
        center: bool = True,
        allow_scale_up: bool = True,
    ):
        self.size = target_size
        self.mode = mode
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.center = center
        self.allow_scale_up = allow_scale_up

    def __call__(self, image_payload: ImagePayload) -> tuple[ImagePayload, ResizeTransform]:
        import cv2

        self._validate_image_payload(image_payload)
        image = image_payload.array
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.size
        interpolation = self._resolve_interpolation(cv2)

        if self.mode == "letterbox":
            ratio = min(target_h / original_h, target_w / original_w)
            if not self.allow_scale_up:
                ratio = min(ratio, 1.0)

            resized_w = int(round(original_w * ratio))
            resized_h = int(round(original_h * ratio))
            resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

            dw = target_w - resized_w
            dh = target_h - resized_h
            if self.center:
                left = int(np.floor(dw / 2))
                right = int(np.ceil(dw / 2))
                top = int(np.floor(dh / 2))
                bottom = int(np.ceil(dh / 2))
            else:
                left = 0
                top = 0
                right = int(dw)
                bottom = int(dh)

            resized = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(self.pad_value, self.pad_value, self.pad_value),
            )
            scale = (ratio, ratio)
            pad = (float(left), float(top))
        elif self.mode == "resize":
            resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
            scale = (target_w / original_w, target_h / original_h)
            pad = (0.0, 0.0)
        else:
            raise ValueError(f"Unsupported resize mode: {self.mode}")

        transform = ResizeTransform(
            scale=scale,
            pad=pad,
            original_shape=(original_h, original_w),
            resized_shape=resized.shape[:2],
        )
        payload = ImagePayload(
            array=resized,
            color_space=image_payload.color_space,
            layout=image_payload.layout,
        )
        return payload, transform

    @staticmethod
    def _validate_image_payload(payload: ImagePayload) -> None:
        if payload.layout != "HWC":
            raise ValueError(f"Resize expects HWC image layout, got {payload.layout}")

    def _resolve_interpolation(self, cv2: object) -> int:
        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        return mapping[self.interpolation]


@Operator
class ConvertColorSpace:
    def __init__(self, output_color_space: Literal["RGB", "BGR"]):
        if output_color_space not in {"RGB", "BGR"}:
            raise ValueError(f"ConvertColorSpace only supports RGB or BGR output, got {output_color_space}")
        self.output_color_space = output_color_space

    def __call__(self, image_payload: ImagePayload) -> ImagePayload:
        if "C" not in image_payload.layout:
            raise ValueError(f"ConvertColorSpace expects a layout containing C, got {image_payload.layout}")

        if image_payload.color_space not in {"RGB", "BGR"}:
            raise ValueError(
                f"ConvertColorSpace only supports BGR/RGB input, got {image_payload.color_space}"
            )

        channel_axis = image_payload.layout.index("C")
        channels = image_payload.channels
        if channels != 3:
            raise ValueError(
                f"ConvertColorSpace only supports 3-channel images, got {channels} for layout {image_payload.layout}"
            )

        array = image_payload.array
        if image_payload.color_space != self.output_color_space:
            array = np.flip(array, axis=channel_axis)
        converted = np.ascontiguousarray(array)
        return ImagePayload(
            array=converted,
            color_space=self.output_color_space,
            layout=image_payload.layout,
        )


@Operator
class Normalize:
    def __init__(
        self,
        scale: float = 1.0 / 255.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        output_layout: Literal["NCHW", "NHWC", "CHW", "HWC"] = "NCHW",
        output_color_space: Literal["RGB", "BGR"] = "RGB",
        add_batch_dim: bool = True,
    ):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.output_layout = output_layout
        self.output_color_space = output_color_space
        self.add_batch_dim = add_batch_dim

    def __call__(self, image_payload: ImagePayload) -> TensorPayload:
        if image_payload.layout != "HWC":
            raise ValueError(f"Normalize expects HWC image layout, got {image_payload.layout}")

        image = image_payload.array
        if image_payload.color_space != self.output_color_space and {
            image_payload.color_space,
            self.output_color_space,
        } == {"BGR", "RGB"}:
            image = image[..., ::-1]
        elif image_payload.color_space != self.output_color_space:
            raise ValueError(
                f"Normalize cannot convert {image_payload.color_space} to {self.output_color_space}"
            )

        if np.issubdtype(image.dtype, np.floating):
            tensor = image.copy()
        else:
            tensor = image.astype(np.float32)
        tensor = tensor * self.scale
        if self.mean is not None:
            tensor = tensor - np.asarray(self.mean, dtype=tensor.dtype)
        if self.std is not None:
            tensor = tensor / np.asarray(self.std, dtype=tensor.dtype)

        if self.output_layout in {"NCHW", "CHW"}:
            tensor = np.transpose(tensor, (2, 0, 1))
        elif self.output_layout not in {"NHWC", "HWC"}:
            raise ValueError(f"Unsupported output layout: {self.output_layout}")

        final_layout = self.output_layout
        if self.add_batch_dim:
            tensor = np.expand_dims(tensor, axis=0)
            if self.output_layout in {"CHW", "HWC"}:
                final_layout = f"N{self.output_layout}"

        return TensorPayload(array=tensor, layout=final_layout, dtype=str(tensor.dtype))


BoxFormat = Literal["xyxy", "xywh", "cxcywh"]
_BOX_FORMATS: frozenset[str] = frozenset(get_args(BoxFormat))


@Operator
class ConvertBoxFormat:
    def __init__(self, src: str = "boxes", *, from_: BoxFormat, to: BoxFormat = "xyxy", as_: str | None = None):
        if from_ not in _BOX_FORMATS:
            raise ValueError(f"ConvertBoxFormat: unknown from_ format {from_!r}. Choose from {sorted(_BOX_FORMATS)}")
        if to not in _BOX_FORMATS:
            raise ValueError(f"ConvertBoxFormat: unknown to format {to!r}. Choose from {sorted(_BOX_FORMATS)}")
        self.src = src
        self.from_ = from_
        self.to = to
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        boxes = registry[self.src]
        registry[self.as_] = self._convert(boxes, self.from_, self.to)
        return registry

    @staticmethod
    def _convert(boxes: np.ndarray, from_: str, to: str) -> np.ndarray:
        if from_ == to:
            return boxes

        if from_ == "xyxy":
            xyxy = boxes
        elif from_ == "xywh":
            xyxy = np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], axis=1)
        elif from_ == "cxcywh":
            half = boxes[:, 2:4] / 2.0
            xyxy = np.concatenate([boxes[:, :2] - half, boxes[:, :2] + half], axis=1)
        else:
            raise ValueError(from_)

        if to == "xyxy":
            return xyxy.astype(boxes.dtype)
        if to == "xywh":
            return np.concatenate([xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2]], axis=1).astype(boxes.dtype)
        if to == "cxcywh":
            wh = xyxy[:, 2:4] - xyxy[:, :2]
            return np.concatenate([xyxy[:, :2] + wh / 2.0, wh], axis=1).astype(boxes.dtype)
        raise ValueError(to)


@Operator
class NMS:
    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        kept_as: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.kept_as = kept_as
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        boxes = registry[self.boxes]
        scores = registry[self.scores]
        classes = registry[self.classes]

        conf_mask = scores >= self.conf_threshold
        filtered_boxes = boxes[conf_mask]
        filtered_scores = scores[conf_mask]
        filtered_classes = classes[conf_mask]
        original_indices = np.where(conf_mask)[0]

        if filtered_boxes.size == 0:
            kept_original = np.zeros((0,), dtype=np.int32)
        else:
            kept_filtered = self._nms_indices(filtered_boxes, filtered_scores, filtered_classes)
            kept_original = original_indices[kept_filtered]

        registry[self.boxes] = boxes[kept_original]
        registry[self.scores] = scores[kept_original]
        registry[self.classes] = classes[kept_original]
        if self.kept_as is not None:
            registry[self.kept_as] = kept_original
        return registry

    def _nms_indices(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        kept_indices: list[int] = []
        for class_id in np.unique(classes):
            class_indices = np.where(classes == class_id)[0]
            ordered = class_indices[np.argsort(scores[class_indices])[::-1]]

            while ordered.size > 0:
                current = int(ordered[0])
                kept_indices.append(current)
                if len(kept_indices) >= self.max_detections or ordered.size == 1:
                    break
                remaining = ordered[1:]
                ious = self._compute_iou(boxes[current], boxes[remaining])
                ordered = remaining[ious < self.iou_threshold]

            if len(kept_indices) >= self.max_detections:
                break

        if not kept_indices:
            return np.zeros((0,), dtype=np.int32)

        kept = np.asarray(kept_indices, dtype=np.int32)
        final_order = np.argsort(scores[kept])[::-1]
        return kept[final_order][: self.max_detections]

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
        intersection = inter_w * inter_h

        box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
        boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
            boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
        )
        union = np.clip(box_area + boxes_area - intersection, a_min=1e-9, a_max=None)
        return intersection / union


@Operator
class NMM:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def __call__(self, detections: Detections) -> Detections:
        if not detections.boxes:
            return detections

        boxes = np.array(detections.boxes, dtype=np.float32)
        scores = np.array(detections.scores, dtype=np.float32)
        classes = np.array(detections.classes, dtype=np.int32)

        merged_boxes: list[list[float]] = []
        merged_scores: list[float] = []
        merged_classes: list[int] = []

        for class_id in np.unique(classes):
            idx = np.where(classes == class_id)[0]
            ordered = idx[np.argsort(scores[idx])[::-1]]
            consumed = np.zeros(len(ordered), dtype=bool)

            for i, current in enumerate(ordered):
                if consumed[i]:
                    continue
                remaining_mask = ~consumed
                remaining_mask[i] = False
                remaining = ordered[remaining_mask]
                if remaining.size > 0:
                    ious = NMS._compute_iou(boxes[current], boxes[remaining])
                    overlap_pos = np.where(ious >= self.iou_threshold)[0]
                    group = np.array([current, *remaining[overlap_pos]])
                    for pos in overlap_pos:
                        consumed[np.where(ordered == remaining[pos])[0]] = True
                else:
                    group = np.array([current])

                group_boxes = boxes[group]
                group_scores = scores[group]
                weights = group_scores / group_scores.sum()
                merged_box = (group_boxes * weights[:, None]).sum(axis=0).tolist()
                merged_boxes.append(merged_box)
                merged_scores.append(float(detections.scores[int(current)]))
                merged_classes.append(int(class_id))

        return Detections(boxes=merged_boxes, scores=merged_scores, classes=merged_classes)


@Operator
class FilterTensorsByScore:
    def __init__(
        self,
        *srcs: str,
        score: str,
        min_score: float,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (score,) + tuple(s for s in srcs if s != score)
        self._inner = FilterTensors(
            *all_srcs,
            by=score,
            predicate=lambda scores: scores >= min_score,
            as_=as_,
        )

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        return self._inner(registry)


@Operator
class FilterTensorsByClasses:
    def __init__(
        self,
        *srcs: str,
        classes: str = "classes",
        keep_classes: Collection[int],
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (classes,) + tuple(src for src in srcs if src != classes)
        allowed_classes = tuple(keep_classes)
        self._inner = FilterTensors(
            *all_srcs,
            by=classes,
            predicate=lambda values: np.isin(values, allowed_classes),
            as_=as_,
        )

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        return self._inner(registry)


@Operator
class FilterTensorsByMasksArea:
    def __init__(
        self,
        *srcs: str,
        masks: str = "masks",
        min_area: int = 1,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (masks,) + tuple(src for src in srcs if src != masks)
        self._inner = FilterTensors(
            *all_srcs,
            by=masks,
            predicate=lambda tensor: _flatten_leading_dim(np.asarray(tensor, dtype=bool)).sum(axis=1) >= min_area,
            as_=as_,
        )

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        return self._inner(registry)


@Operator
class WeightMasksByScores:
    def __init__(self, masks: str = "masks", scores: str = "scores", *, as_: str):
        self.masks = masks
        self.scores = scores
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        scores = registry[self.scores]
        masks = registry[self.masks]
        expanded_scores = scores.reshape((scores.shape[0],) + (1,) * (masks.ndim - 1))
        registry[self.as_] = expanded_scores * masks
        return registry


@Operator
class ResizeMasks:
    def __init__(self, masks: str = "masks", as_: str | None = None):
        self.masks = masks
        self.as_ = as_ or masks

    def __call__(self, registry: TensorRegistry, image_shape: tuple[int, int]) -> TensorRegistry:
        import cv2

        height, width = image_shape
        masks = registry[self.masks]
        resized = [cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) for mask in masks]
        registry[self.as_] = (
            np.stack(resized, axis=0)
            if resized
            else np.zeros((0, height, width), dtype=masks.dtype)
        )
        return registry


@Operator
class MeanMaskScores:
    def __init__(
        self,
        masks: str = "masks",
        binary_masks: str | None = "binary_masks",
        *,
        as_: str,
    ):
        self.masks = masks
        self.binary_masks = binary_masks
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        masks = registry[self.masks]
        if self.binary_masks is None:
            registry[self.as_] = _flatten_leading_dim(masks).mean(axis=1)
            return registry

        binary_masks = registry[self.binary_masks]
        areas = _flatten_leading_dim(binary_masks).sum(axis=1)
        mask_sums = _flatten_leading_dim(masks * binary_masks).sum(axis=1)
        registry[self.as_] = np.where(areas > 0, mask_sums / np.clip(areas, 1, None), 0.0)
        return registry


@Operator
class MasksToBoxes:
    def __init__(self, masks: str = "masks", *, as_: str):
        self.masks = masks
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        masks = registry[self.masks]
        count = masks.shape[0]
        if count == 0:
            registry[self.as_] = np.zeros((0, 4), dtype=np.float32)
            return registry

        _, height, width = masks.shape
        xs = np.arange(width, dtype=np.float32).reshape(1, 1, width)
        ys = np.arange(height, dtype=np.float32).reshape(1, height, 1)
        x1 = np.where(masks, xs, float(width)).min(axis=(-2, -1))
        y1 = np.where(masks, ys, float(height)).min(axis=(-2, -1))
        x2 = np.where(masks, xs, -1.0).max(axis=(-2, -1)) + 1.0
        y2 = np.where(masks, ys, -1.0).max(axis=(-2, -1)) + 1.0
        boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32, copy=False)
        empty = ~masks.any(axis=(-2, -1))
        boxes[empty] = 0.0
        registry[self.as_] = boxes
        return registry


@Operator
class ReconstructMasks:
    def __init__(self, coefficients: str, prototypes: str, as_: str):
        self.coefficients = coefficients
        self.prototypes = prototypes
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        coefficients = registry[self.coefficients]
        prototypes = registry[self.prototypes]
        channels, mask_h, mask_w = prototypes.shape
        masks = coefficients @ prototypes.reshape(channels, -1)
        registry[self.as_] = masks.reshape(-1, mask_h, mask_w)
        return registry


@Operator
class ProjectBoxes:
    def __init__(self, src: str = "boxes"):
        self.src = src

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        boxes = registry[self.src].copy()
        pad_x, pad_y = transform.pad
        scale_x, scale_y = transform.scale
        original_h, original_w = transform.original_shape

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))
        registry[self.src] = boxes
        return registry


@Operator
class ProjectMasks:
    def __init__(self, masks: str = "masks", boxes: str = "boxes", mask_threshold: float = 0.5):
        self.masks = masks
        self.boxes = boxes
        self.mask_threshold = mask_threshold

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        import cv2

        masks = registry[self.masks]
        boxes = registry[self.boxes]
        count = masks.shape[0]
        resized_h, resized_w = transform.resized_shape
        orig_h, orig_w = transform.original_shape
        scale_x, scale_y = transform.scale
        pad_x, pad_y = transform.pad
        _, proto_h, proto_w = masks.shape

        proto_boxes = boxes.astype(np.float32).copy()
        proto_x = (proto_boxes[:, [0, 2]] * scale_x + pad_x) * (proto_w / resized_w)
        proto_boxes[:, [0, 2]] = proto_x

        proto_y = (proto_boxes[:, [1, 3]] * scale_y + pad_y) * (proto_h / resized_h)
        proto_boxes[:, [1, 3]] = proto_y

        x1 = proto_boxes[:, 0].clip(0, proto_w)[:, None, None]
        y1 = proto_boxes[:, 1].clip(0, proto_h)[:, None, None]
        x2 = proto_boxes[:, 2].clip(0, proto_w)[:, None, None]
        y2 = proto_boxes[:, 3].clip(0, proto_h)[:, None, None]
        cols = np.arange(proto_w, dtype=np.float32)[None, None, :]
        rows = np.arange(proto_h, dtype=np.float32)[None, :, None]
        inside_cols = (cols >= x1) & (cols < x2)
        inside_rows = (rows >= y1) & (rows < y2)
        masks = masks * (inside_cols & inside_rows)

        top = max(int(round(pad_y - 0.1)), 0)
        left = max(int(round(pad_x - 0.1)), 0)
        bottom = min(int(round(resized_h - pad_y + 0.1)), resized_h)
        right = min(int(round(resized_w - pad_x + 0.1)), resized_w)

        projected = np.empty((count, orig_h, orig_w), dtype=np.uint8)
        for index, mask in enumerate(masks):
            upsampled = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            cropped = upsampled[top:bottom, left:right]
            resized = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            projected[index] = (resized > self.mask_threshold).astype(np.uint8)

        registry[self.masks] = projected
        return registry


@Operator
class ProjectRoIMasks:
    def __init__(self, masks: str = "masks", boxes: str = "boxes", mask_threshold: float = 0.5):
        self.masks = masks
        self.boxes = boxes
        self.mask_threshold = mask_threshold

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        import cv2

        boxes = registry[self.boxes]
        masks = registry[self.masks]
        orig_h, orig_w = transform.original_shape

        canvas = np.zeros((len(boxes), orig_h, orig_w), dtype=bool)
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1 = max(0, int(np.floor(box[0])))
            y1 = max(0, int(np.floor(box[1])))
            x2 = min(orig_w, int(np.ceil(box[2])))
            y2 = min(orig_h, int(np.ceil(box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            resized = cv2.resize(mask.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            canvas[i, y1:y2, x1:x2] = resized > self.mask_threshold

        registry[self.masks] = canvas
        return registry


@Operator
class ToDetections:
    def __init__(self, boxes: str = "boxes", scores: str = "scores", classes: str = "classes"):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    def __call__(self, registry: TensorRegistry) -> Detections:
        return Detections(
            boxes=registry[self.boxes].tolist(),
            scores=registry[self.scores].tolist(),
            classes=registry[self.classes].tolist(),
        )


@Operator
class ToSegmentations:
    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        masks: str = "masks",
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.masks = masks

    def __call__(self, registry: TensorRegistry) -> Segmentations:
        return Segmentations(
            boxes=registry[self.boxes].tolist(),
            scores=registry[self.scores].tolist(),
            classes=registry[self.classes].tolist(),
            masks=list(registry[self.masks]),
        )


@Operator
class FilterPredictions(Generic[PredictionT]):
    def __init__(self, predicate: Callable[[PredictionT], PredictionMask]):
        self.predicate = predicate

    def __call__(self, prediction: PredictionT) -> PredictionT:
        return prediction.filter(self.predicate(prediction))


@Operator
class FilterPredictionsByClass:
    def __init__(self, classes: Collection[int]):
        self.classes = frozenset(classes)

    def __call__(self, prediction: ClassPredictionT) -> ClassPredictionT:
        return prediction.filter([class_id in self.classes for class_id in prediction.classes])


@Operator
class FilterPredictionsByScore:
    def __init__(self, min_score: float):
        self.min_score = min_score

    def __call__(self, prediction: ScorePredictionT) -> ScorePredictionT:
        return prediction.filter([score >= self.min_score for score in prediction.scores])


@Operator
class FilterPredictionsByArea:
    def __init__(self, min_area: float = 0, max_area: float | None = None):
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, prediction: BoxPredictionT) -> BoxPredictionT:
        return prediction.filter([
            (x2 - x1) * (y2 - y1) >= self.min_area
            and (self.max_area is None or (x2 - x1) * (y2 - y1) <= self.max_area)
            for x1, y1, x2, y2 in prediction.boxes
        ])


@Operator
class DrawBoxes:
    def __init__(
        self,
        class_names: list[str] | tuple[str, ...] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.class_names = tuple(class_names) if class_names is not None else None
        self.color = color
        self.thickness = thickness
        self.font_scale = font_scale

    def __call__(self, source_image: ImagePayload, detections: DetectT) -> tuple[ImagePayload, DetectT]:
        import cv2

        image = source_image.array.copy()
        for box, score, class_id in zip(detections.boxes, detections.scores, detections.classes, strict=True):
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, self.thickness)
            label = self._format_label(int(class_id), float(score))
            text_origin_y = y1 - 8 if y1 > 18 else y1 + 18
            cv2.putText(
                image,
                label,
                (x1, text_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                max(1, self.thickness - 1),
                cv2.LINE_AA,
            )

        return ImagePayload(array=image, color_space="BGR", layout="HWC"), detections

    def _format_label(self, class_id: int, score: float) -> str:
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = str(class_id)
        return f"{name} {score:.2f}"


@Operator
class DrawMasks:
    def __init__(
        self,
        class_names: list[str] | tuple[str, ...] | None = None,
        alpha: float = 0.45,
    ):
        self.class_names = tuple(class_names) if class_names is not None else None
        self.alpha = alpha

    def __call__(self, source_image: ImagePayload, segmentations: SegT) -> tuple[ImagePayload, SegT]:
        image = source_image.array.copy()
        for mask, class_id in zip(segmentations.masks, segmentations.classes, strict=True):
            color = np.asarray(self._class_color(int(class_id)), dtype=np.float32)
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.ndim != 2:
                raise ValueError(f"Expected 2D segmentation mask, got shape {mask_bool.shape}")
            if np.any(mask_bool):
                blended = (1.0 - self.alpha) * image[mask_bool].astype(np.float32) + self.alpha * color
                image[mask_bool] = blended.astype(np.uint8)

        return ImagePayload(array=image, color_space=source_image.color_space, layout=source_image.layout), segmentations

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        return (
            int((37 * class_id + 17) % 255),
            int((91 * class_id + 53) % 255),
            int((17 * class_id + 191) % 255),
        )


@Operator
class SaveImage(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(self, output_path: str | Path, at: int | None = None):
        self.output_path = Path(output_path)
        self.at = at

    def effect(self, payload: PayloadT) -> None:
        import cv2

        payload_value: Any = payload
        image_payload = payload_value[self.at] if self.at is not None else payload_value
        if image_payload.layout != "HWC":
            raise ValueError(f"SaveImage expects HWC image layout, got {image_payload.layout}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(self.output_path), image_payload.array)
        if not written:
            raise ValueError(f"Failed to write image: {self.output_path}")


@Operator
class MapPredictionsToObjects(Generic[ObjectIndexT, PredictionT]):
    @overload
    def __init__(
        self: "MapPredictionsToObjects[None, PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "MapPredictionsToObjects[Literal[1], PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: Literal[1],
    ) -> None:
        ...

    @overload
    def __init__(
        self: "MapPredictionsToObjects[int, PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: int,
    ) -> None:
        ...

    def __init__(
        self,
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: int | None = None,
    ) -> None:
        self.fields = fields
        self.at = at

    @overload
    def __call__(
        self: "MapPredictionsToObjects[None, PredictionT]",
        payload: PredictionT,
    ) -> list[ObjectMapping]:
        ...

    @overload
    def __call__(
        self: "MapPredictionsToObjects[Literal[1], PredictionT]",
        payload: tuple[ObjectPrefixT, PredictionT],
    ) -> tuple[ObjectPrefixT, list[ObjectMapping]]:
        ...

    @overload
    def __call__(self, payload: object) -> Any:
        ...

    def __call__(self, payload: object) -> Any:
        prediction_arrays = self._resolve_prediction_value(payload)
        columns: dict[str, Sequence[object]] = {}
        for field_name, source in self.fields.items():
            if isinstance(source, str):
                try:
                    column = getattr(prediction_arrays, source)
                except AttributeError as exc:
                    raise AttributeError(
                        f"MapPredictionsToObjects field {field_name!r} references missing attribute "
                        f"{source!r} on {type(prediction_arrays).__name__}"
                    ) from exc
            else:
                column = source(prediction_arrays)
            columns[field_name] = column

        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"MapPredictionsToObjects requires equal-length collections, got lengths {sorted(lengths)}"
            )

        records: list[dict[str, object]] = []
        field_names = tuple(columns.keys())
        rows = zip(*(columns[field_name] for field_name in field_names), strict=True)
        for row in rows:
            records.append(dict(zip(field_names, row, strict=True)))
        if self.at is not None:
            payload_tuple = cast(tuple[object, ...], payload)
            return payload_tuple[:self.at] + (records,) + payload_tuple[self.at + 1 :]
        return records

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        if self.at is None:
            if current_output is not Any and is_assignable(current_output, Prediction):
                return (current_output,), list[ObjectMapping]
            return (Any,), Any

        if current_output is Any:
            return (Any,), Any

        if get_origin(current_output) is tuple:
            parts = get_args(current_output)
        elif isinstance(current_output, tuple):
            parts = current_output
        else:
            return (Any,), Any

        normalized_index = self.at if self.at >= 0 else len(parts) + self.at
        if normalized_index < 0 or normalized_index >= len(parts):
            error_type = validation_error_type or PipelineValidationError
            raise error_type(
                f"MapPredictionsToObjects(at={self.at}) is out of bounds for "
                f"{current_output} (length {len(parts)})"
            )
        if not is_assignable(parts[normalized_index], Prediction):
            return (Any,), Any
        updated_parts = parts[:normalized_index] + (list[ObjectMapping],) + parts[normalized_index + 1 :]
        return (current_output,), updated_parts

    def _resolve_prediction_value(self, payload: object) -> PredictionT:
        if self.at is None:
            return cast(PredictionT, payload)
        if not isinstance(payload, tuple):
            raise TypeError(
                f"MapPredictionsToObjects(at={self.at}) requires a tuple payload, got {type(payload)!r}"
            )
        return cast(PredictionT, payload[self.at])


@Operator
class LogDetections(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(
        self,
        model_path: str | Path,
        image_path: str | Path,
        annotated_image_path: str | Path,
        indent: int = 2,
        stream: TextIO | None = None,
        at: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.image_path = Path(image_path)
        self.annotated_image_path = Path(annotated_image_path)
        self.indent = indent
        self.stream = stream or sys.stdout
        self.at = at

    def effect(self, payload: PayloadT) -> None:
        payload_value: Any = payload
        prediction_objects = payload_value[self.at] if self.at is not None else payload_value
        print(
            json.dumps(
                {
                    "model": str(self.model_path),
                    "image": str(self.image_path),
                    "annotated_image": str(self.annotated_image_path),
                    "detections": prediction_objects,
                },
                indent=self.indent,
            ),
            file=self.stream,
        )

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation

        concrete_objects = list[ObjectMapping]
        if self.at is None:
            if current_output is Any or _is_unresolved_object_list(current_output):
                return (concrete_objects,), concrete_objects
            return (current_output,), current_output

        if current_output is Any:
            return (Any,), Any

        if get_origin(current_output) is tuple:
            parts = get_args(current_output)
        elif isinstance(current_output, tuple):
            parts = current_output
        else:
            return (Any,), Any

        normalized_index = self.at if self.at >= 0 else len(parts) + self.at
        if normalized_index < 0 or normalized_index >= len(parts):
            error_type = validation_error_type or PipelineValidationError
            raise error_type(
                f"LogDetections(at={self.at}) is out of bounds for "
                f"{current_output} (length {len(parts)})"
            )

        logged_annotation = parts[normalized_index]
        if logged_annotation is Any or _is_unresolved_object_list(logged_annotation):
            logged_annotation = concrete_objects
        updated_parts = parts[:normalized_index] + (logged_annotation,) + parts[normalized_index + 1 :]
        return (current_output,), updated_parts
