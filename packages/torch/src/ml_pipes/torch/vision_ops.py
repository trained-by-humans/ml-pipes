from __future__ import annotations

from collections.abc import Collection
from typing import Literal, get_args

import torch

from ml_pipes.operator import Operator
from .tensor_ops import TorchFilterTensors
from .types import TorchTensorRegistry

__all__ = [
    "TorchConvertBoxFormat",
    "TorchFilterTensorsByScore",
    "TorchFilterTensorsByClasses",
    "TorchFilterTensorsByMasksArea",
    "TorchWeightMasksByScores",
    "TorchResizeMasks",
    "TorchMeanMaskScores",
    "TorchMasksToBoxes",
    "TorchReconstructMasks",
    "TorchNMS",
]

BoxFormat = Literal["xyxy", "xywh", "cxcywh"]
_BOX_FORMATS: frozenset[str] = frozenset(get_args(BoxFormat))


@Operator
class TorchConvertBoxFormat:
    def __init__(
        self,
        src: str = "boxes",
        *,
        from_: BoxFormat,
        to: BoxFormat = "xyxy",
        as_: str | None = None,
    ):
        if from_ not in _BOX_FORMATS:
            raise ValueError(
                f"TorchConvertBoxFormat: unknown from_ format {from_!r}. Choose from {sorted(_BOX_FORMATS)}"
            )
        if to not in _BOX_FORMATS:
            raise ValueError(f"TorchConvertBoxFormat: unknown to format {to!r}. Choose from {sorted(_BOX_FORMATS)}")
        self.src = src
        self.from_ = from_
        self.to = to
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        boxes = registry[self.src]
        registry[self.as_] = self._convert(boxes, self.from_, self.to)
        return registry

    @staticmethod
    def _convert(boxes: torch.Tensor, from_: str, to: str) -> torch.Tensor:
        if from_ == to:
            return boxes

        if from_ == "xyxy":
            xyxy = boxes
        elif from_ == "xywh":
            xyxy = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], dim=1)
        elif from_ == "cxcywh":
            half = boxes[:, 2:4] / 2.0
            xyxy = torch.cat([boxes[:, :2] - half, boxes[:, :2] + half], dim=1)
        else:
            raise ValueError(from_)

        if to == "xyxy":
            return xyxy.to(dtype=boxes.dtype)
        if to == "xywh":
            return torch.cat([xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2]], dim=1).to(dtype=boxes.dtype)
        if to == "cxcywh":
            wh = xyxy[:, 2:4] - xyxy[:, :2]
            return torch.cat([xyxy[:, :2] + wh / 2.0, wh], dim=1).to(dtype=boxes.dtype)
        raise ValueError(to)


@Operator
class TorchFilterTensorsByScore:
    def __init__(
        self,
        *srcs: str,
        score: str,
        min_score: float,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (score,) + tuple(src for src in srcs if src != score)
        self._inner = TorchFilterTensors(
            *all_srcs,
            by=score,
            predicate=lambda scores: scores >= min_score,
            as_=as_,
        )

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


@Operator
class TorchFilterTensorsByClasses:
    """Filters tensors by class id membership."""

    def __init__(
        self,
        *srcs: str,
        classes: str = "classes",
        keep_classes: Collection[int],
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (classes,) + tuple(src for src in srcs if src != classes)
        allowed = tuple(keep_classes)
        self._inner = TorchFilterTensors(
            *all_srcs,
            by=classes,
            predicate=lambda values: torch.isin(
                values,
                torch.as_tensor(allowed, device=values.device, dtype=values.dtype),
            ),
            as_=as_,
        )

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


@Operator
class TorchFilterTensorsByMasksArea:
    def __init__(
        self,
        *srcs: str,
        masks: str = "masks",
        min_area: int = 1,
        as_: str | tuple[str, ...] | None = None,
    ):
        all_srcs = (masks,) + tuple(src for src in srcs if src != masks)
        self._inner = TorchFilterTensors(
            *all_srcs,
            by=masks,
            predicate=lambda tensor: tensor.to(dtype=torch.bool).flatten(1).sum(dim=1) >= min_area,
            as_=as_,
        )

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        return self._inner(registry)


@Operator
class TorchWeightMasksByScores:
    def __init__(self, masks: str = "masks", scores: str = "scores", *, as_: str):
        self.masks = masks
        self.scores = scores
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        scores = registry[self.scores]
        masks = registry[self.masks]
        expanded_scores = scores.reshape((scores.shape[0],) + (1,) * (masks.ndim - 1))
        registry[self.as_] = expanded_scores * masks
        return registry


@Operator
class TorchResizeMasks:
    """Resizes a stack of masks to a target shape."""

    def __init__(self, masks: str = "masks", as_: str | None = None):
        self.masks = masks
        self.as_ = as_ or masks

    def __call__(self, registry: TorchTensorRegistry, image_shape: tuple[int, int]) -> TorchTensorRegistry:
        masks = registry[self.masks]
        resized = torch.nn.functional.interpolate(
            masks[:, None, :, :],
            size=image_shape,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        registry[self.as_] = resized
        return registry


@Operator
class TorchMeanMaskScores:
    """Computes one mean score per mask from dense mask values.

    If `binary_masks` is provided, the mean is computed only over pixels where
    the binary mask is True. If `binary_masks` is None, the mean is computed
    over all pixels in each dense mask.
    """

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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        if self.binary_masks is None:
            registry[self.as_] = masks.flatten(1).mean(dim=1)
            return registry

        binary_masks = registry[self.binary_masks]
        areas = binary_masks.flatten(1).sum(dim=1)
        registry[self.as_] = torch.where(
            areas > 0,
            (masks * binary_masks).flatten(1).sum(dim=1) / areas.clamp_min(1).to(masks.dtype),
            torch.zeros((masks.shape[0],), dtype=masks.dtype, device=masks.device),
        )
        return registry


@Operator
class TorchMasksToBoxes:
    def __init__(self, masks: str = "masks", *, as_: str):
        self.masks = masks
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        masks = registry[self.masks]
        count = masks.shape[0]
        if count == 0:
            registry[self.as_] = torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
            return registry

        _, height, width = masks.shape
        xs = torch.arange(width, dtype=torch.float32, device=masks.device).view(1, 1, width)
        ys = torch.arange(height, dtype=torch.float32, device=masks.device).view(1, height, 1)
        x1 = torch.where(masks, xs, float(width)).amin(dim=(-2, -1))
        y1 = torch.where(masks, ys, float(height)).amin(dim=(-2, -1))
        x2 = torch.where(masks, xs, -1.0).amax(dim=(-2, -1)) + 1.0
        y2 = torch.where(masks, ys, -1.0).amax(dim=(-2, -1)) + 1.0
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        empty = ~masks.any(dim=(-2, -1))
        boxes[empty] = 0.0
        registry[self.as_] = boxes
        return registry


@Operator
class TorchReconstructMasks:
    def __init__(self, coefficients: str, prototypes: str, as_: str):
        self.coefficients = coefficients
        self.prototypes = prototypes
        self.as_ = as_

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        coefficients = registry[self.coefficients]
        prototypes = registry[self.prototypes]
        channels, mask_h, mask_w = prototypes.shape
        masks = coefficients @ prototypes.reshape(channels, -1)
        registry[self.as_] = masks.reshape(-1, mask_h, mask_w)
        return registry


@Operator
class TorchNMS:
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        boxes = registry[self.boxes]
        scores = registry[self.scores]
        classes = registry[self.classes]

        conf_mask = scores >= self.conf_threshold
        filtered_boxes = boxes[conf_mask]
        filtered_scores = scores[conf_mask]
        filtered_classes = classes[conf_mask]
        original_indices = torch.nonzero(conf_mask, as_tuple=False).squeeze(1)

        if filtered_boxes.numel() == 0:
            kept_original = torch.zeros((0,), dtype=torch.int64, device=boxes.device)
        else:
            kept_filtered = self._nms_indices(filtered_boxes, filtered_scores, filtered_classes)
            kept_original = original_indices[kept_filtered]

        registry[self.boxes] = boxes[kept_original]
        registry[self.scores] = scores[kept_original]
        registry[self.classes] = classes[kept_original]
        if self.kept_as is not None:
            registry[self.kept_as] = kept_original.to(dtype=torch.int64)
        return registry

    def _nms_indices(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
    ) -> torch.Tensor:
        from torchvision.ops import nms

        kept_groups: list[torch.Tensor] = []
        for class_id in torch.unique(classes):
            class_indices = torch.nonzero(classes == class_id, as_tuple=False).squeeze(1)
            if class_indices.numel() == 0:
                continue
            kept_groups.append(class_indices[nms(boxes[class_indices], scores[class_indices], self.iou_threshold)])

        if not kept_groups:
            return torch.zeros((0,), dtype=torch.int64, device=boxes.device)

        kept = torch.cat(kept_groups)
        ordered = kept[torch.argsort(scores[kept], descending=True)]
        return ordered[: self.max_detections].to(dtype=torch.int64)
