from __future__ import annotations

import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.tensor import FilterTensors, TensorRegistry
from .types import ImagePayload, ResizeTransform

__all__ = [
    "DrawMasks",
    "FilterTensorsByMasksArea",
    "MasksToBoxes",
    "MeanMaskedScores",
    "ProjectMasks",
    "ProjectRoIMasks",
    "ReconstructMasks",
    "ResizeMasks",
    "WeightMasksByScores",
]

def _flatten_leading_dim(array: np.ndarray) -> np.ndarray:
    leading = int(array.shape[0])
    trailing = int(np.prod(array.shape[1:], dtype=np.int64))
    return array.reshape(leading, trailing)


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
class MeanMaskedScores:
    """Computes one score per instance by averaging dense mask scores over its mask."""

    def __init__(
        self,
        mask_scores: str = "mask_scores",
        masks: str = "masks",
        *,
        as_: str,
    ):
        self.mask_scores = mask_scores
        self.masks = masks
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        mask_scores = registry[self.mask_scores]
        masks = registry[self.masks]
        areas = _flatten_leading_dim(masks).sum(axis=1)
        mask_sums = _flatten_leading_dim(mask_scores * masks).sum(axis=1)
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
class DrawMasks:
    def __init__(
        self,
        masks: str = "masks",
        classes: str = "classes",
        *,
        class_names: list[str] | tuple[str, ...] | None = None,
        alpha: float = 0.45,
    ):
        self.class_names = tuple(class_names) if class_names is not None else None
        self.alpha = alpha
        self.masks = masks
        self.classes = classes

    def __call__(self, source_image: ImagePayload, registry: TensorRegistry) -> tuple[ImagePayload, TensorRegistry]:
        image = source_image.array.copy()
        for mask, class_id in zip(registry[self.masks], registry[self.classes], strict=True):
            color = np.asarray(self._class_color(int(class_id)), dtype=np.float32)
            if source_image.color_space == "RGB":
                color = color[::-1]
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.ndim != 2:
                raise ValueError(f"Expected 2D segmentation mask, got shape {mask_bool.shape}")
            if np.any(mask_bool):
                blended = (1.0 - self.alpha) * image[mask_bool].astype(np.float32) + self.alpha * color
                image[mask_bool] = blended.astype(np.uint8)

        return ImagePayload(array=image, color_space=source_image.color_space, layout=source_image.layout), registry

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        return (
            int((37 * class_id + 17) % 255),
            int((91 * class_id + 53) % 255),
            int((17 * class_id + 191) % 255),
        )
