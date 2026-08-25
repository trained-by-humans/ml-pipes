from __future__ import annotations

import json
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any, Generic, Literal, TextIO, TypeVar, get_args, get_origin

import numpy as np

from ml_pipes._typing.annotation import is_concrete_annotation
from ml_pipes.operator import Operator
from ml_pipes.standard import SideEffectOp
from ml_pipes.tensor import FilterTensors, TensorRegistry
from ml_pipes.validation import PipelineValidationError
from .types import ImagePayload, ResizeTransform

__all__ = [
    "ConvertBoxFormat",
    "DrawBoxes",
    "FilterTensorsByBoxArea",
    "FilterTensorsByClasses",
    "FilterTensorsByScore",
    "LogDetections",
    "NMM",
    "NMS",
    "ProjectBoxes",
]

PayloadT = TypeVar("PayloadT")


def _is_unresolved_object_list(annotation: Any) -> bool:
    return annotation is list or (get_origin(annotation) is list and not is_concrete_annotation(annotation))


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
        if classes is None:
            raise ValueError("NMS requires a classes tensor")
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
    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        iou_threshold: float = 0.5,
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.iou_threshold = iou_threshold

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        boxes = registry[self.boxes]
        scores = registry[self.scores]
        classes = registry[self.classes]
        if boxes.shape[0] == 0:
            return registry

        computation_dtype = np.result_type(boxes.dtype, scores.dtype, np.float32)
        computation_boxes = np.asarray(boxes, dtype=computation_dtype)
        computation_scores = np.asarray(scores, dtype=computation_dtype)

        merged_boxes: list[list[float]] = []
        merged_scores: list[float] = []
        merged_classes: list[int] = []

        for class_id in np.unique(classes):
            idx = np.where(classes == class_id)[0]
            ordered = idx[np.argsort(computation_scores[idx])[::-1]]
            consumed = np.zeros(len(ordered), dtype=bool)

            for i, current in enumerate(ordered):
                if consumed[i]:
                    continue
                remaining_mask = ~consumed
                remaining_mask[i] = False
                remaining = ordered[remaining_mask]
                if remaining.size > 0:
                    ious = NMS._compute_iou(computation_boxes[current], computation_boxes[remaining])
                    overlap_pos = np.where(ious >= self.iou_threshold)[0]
                    group = np.array([current, *remaining[overlap_pos]])
                    for pos in overlap_pos:
                        consumed[np.where(ordered == remaining[pos])[0]] = True
                else:
                    group = np.array([current])

                group_boxes = computation_boxes[group]
                group_scores = computation_scores[group]
                score_sum = float(group_scores.sum())
                weights = (
                    group_scores / score_sum
                    if score_sum > 0
                    else np.full(group_scores.shape, 1.0 / group_scores.size, dtype=computation_dtype)
                )
                merged_box = (group_boxes * weights[:, None]).sum(axis=0).tolist()
                merged_boxes.append(merged_box)
                merged_scores.append(float(scores[int(current)]))
                merged_classes.append(int(class_id))

        registry[self.boxes] = np.asarray(merged_boxes, dtype=boxes.dtype).reshape(-1, 4)
        registry[self.scores] = np.asarray(merged_scores, dtype=scores.dtype)
        registry[self.classes] = np.asarray(merged_classes, dtype=classes.dtype)
        return registry


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
class FilterTensorsByBoxArea:
    def __init__(
        self,
        *srcs: str,
        boxes: str = "boxes",
        min_area: float = 0,
        max_area: float | None = None,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (boxes,) + tuple(src for src in srcs if src != boxes)

        def keep(values: np.ndarray) -> np.ndarray:
            areas = self._areas(values)
            return areas >= min_area if max_area is None else (areas >= min_area) & (areas <= max_area)

        self._inner = FilterTensors(*all_srcs, by=boxes, predicate=keep, as_=as_)

    @staticmethod
    def _areas(boxes: np.ndarray) -> np.ndarray:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        return self._inner(registry)


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
class DrawBoxes:
    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str | None = "classes",
        *,
        class_names: list[str] | tuple[str, ...] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ):
        if class_names is not None and classes is None:
            raise ValueError("DrawBoxes class_names requires a classes tensor")
        self.class_names = tuple(class_names) if class_names is not None else None
        self.color = color
        self.thickness = thickness
        self.font_scale = font_scale
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    def __call__(self, source_image: ImagePayload, registry: TensorRegistry) -> tuple[ImagePayload, TensorRegistry]:
        import cv2

        image = source_image.array.copy()
        color = self.color[::-1] if source_image.color_space == "RGB" else self.color
        class_ids = registry[self.classes] if self.classes is not None else (None,) * len(registry[self.boxes])
        for box, score, class_id in zip(registry[self.boxes], registry[self.scores], class_ids, strict=True):
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.thickness)
            label = self._format_label(int(class_id), float(score)) if class_id is not None else f"{score:.2f}"
            text_origin_y = y1 - 8 if y1 > 18 else y1 + 18
            cv2.putText(
                image,
                label,
                (x1, text_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                max(1, self.thickness - 1),
                cv2.LINE_AA,
            )

        return ImagePayload(array=image, color_space=source_image.color_space, layout=source_image.layout), registry

    def _format_label(self, class_id: int, score: float) -> str:
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = str(class_id)
        return f"{name} {score:.2f}"


@Operator
class LogDetections(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(
        self,
        model_path: str | Path,
        image_path: str | Path,
        annotated_image_path: str | Path,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        *,
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
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    def effect(self, payload: PayloadT) -> None:
        payload_value: Any = payload
        registry = payload_value[self.at] if self.at is not None else payload_value
        detections = [
            {"box": box.tolist(), "score": float(score), "class_id": int(class_id)}
            for box, score, class_id in zip(
                registry[self.boxes], registry[self.scores], registry[self.classes], strict=True
            )
        ]
        print(
            json.dumps(
                {
                    "model": str(self.model_path),
                    "image": str(self.image_path),
                    "annotated_image": str(self.annotated_image_path),
                    "detections": detections,
                },
                indent=self.indent,
            ),
            file=self.stream,
        )

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if self.at is None:
            return (TensorRegistry,), TensorRegistry

        if upstream_annotation is Any:
            return (Any,), Any

        if get_origin(upstream_annotation) is tuple:
            parts = get_args(upstream_annotation)
        elif isinstance(upstream_annotation, tuple):
            parts = upstream_annotation
        else:
            return (Any,), Any

        normalized_index = self.at if self.at >= 0 else len(parts) + self.at
        if normalized_index < 0 or normalized_index >= len(parts):
            error_type = validation_error_type or PipelineValidationError
            raise error_type(
                f"LogDetections(at={self.at}) is out of bounds for "
                f"{upstream_annotation} (length {len(parts)})"
            )

        return (upstream_annotation,), upstream_annotation
