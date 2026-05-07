from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.torch"

from examples.common import COCO_IMAGE_NAME, COCO_IMAGE_URL, add_assets_dir_arg, download_if_missing, visualize_and_store
from ml_pipes import LogDetections, MapToObjects, Pipeline, Recall, ToSegmentations, inline
from ml_pipes.torch import ToNumpyRegistry
from ml_pipes.types import TensorRegistry

from .mask2former_infer import (
    Mask2FormerInfer,
    LoadedMask2Former,
    build_mask2former_preprocess_pipeline,
    resolve_output_path,
    resolve_task_list,
)


def stable_sigmoid(array: np.ndarray) -> np.ndarray:
    positive = array >= 0
    result = np.empty_like(array, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


class ResizeMasksToImage:
    def __init__(self, image_shape: tuple[int, int], src: str = "masks_queries_logits", as_: str | None = None):
        self.image_shape = image_shape
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        import cv2

        height, width = self.image_shape
        masks = registry[self.src]
        resized = [cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) for mask in masks]
        registry[self.as_] = np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=np.float32)
        return registry


class ResizeMasksToStoredImage:
    def __init__(self, src: str = "masks_queries_logits", as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry, image_shape: tuple[int, int]) -> TensorRegistry:
        return ResizeMasksToImage(image_shape=image_shape, src=self.src, as_=self.as_)(registry)


class ClassProbabilities:
    def __init__(self, src: str = "class_queries_logits", as_: str = "class_probs"):
        self.src = src
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        logits = registry[self.src]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        registry[self.as_] = probs[..., :-1].astype(np.float32, copy=False)
        return registry


class MaskProbabilities:
    def __init__(self, src: str = "masks_queries_logits", as_: str = "mask_probs"):
        self.src = src
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = stable_sigmoid(registry[self.src])
        return registry


def _empty_numpy_segments(image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    return (
        np.zeros((0, height, width), dtype=bool),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def _append_or_merge_numpy_panoptic_segment(
    merged_masks: list[np.ndarray],
    merged_scores: list[float],
    merged_classes: list[int],
    stuff_lookup: dict[int, int],
    class_id: int,
    score: float,
    mask: np.ndarray,
    thing_class_ids: frozenset[int],
) -> None:
    if class_id in thing_class_ids:
        merged_masks.append(mask)
        merged_scores.append(score)
        merged_classes.append(class_id)
        return

    existing = stuff_lookup.get(class_id)
    if existing is None:
        stuff_lookup[class_id] = len(merged_masks)
        merged_masks.append(mask)
        merged_scores.append(score)
        merged_classes.append(class_id)
        return

    merged_masks[existing] = merged_masks[existing] | mask
    merged_scores[existing] = max(merged_scores[existing], score)


def _finalize_numpy_segments(
    merged_masks: list[np.ndarray],
    merged_scores: list[float],
    merged_classes: list[int],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not merged_masks:
        return _empty_numpy_segments(image_shape)
    return (
        np.stack(merged_masks, axis=0).astype(bool, copy=False),
        np.asarray(merged_scores, dtype=np.float32),
        np.asarray(merged_classes, dtype=np.int64),
    )


class PanopticQueryPredictions:
    def __init__(
        self,
        class_probs: str = "class_probs",
        scores_as: str = "query_scores",
        classes_as: str = "query_classes",
    ):
        self.class_probs = class_probs
        self.scores_as = scores_as
        self.classes_as = classes_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        class_probs = registry[self.class_probs]
        registry[self.scores_as] = class_probs.max(axis=-1).astype(np.float32, copy=False)
        registry[self.classes_as] = class_probs.argmax(axis=-1).astype(np.int64, copy=False)
        return registry


class PanopticKeepQueries:
    def __init__(
        self,
        score_threshold: float = 0.5,
        scores: str = "query_scores",
        classes: str = "query_classes",
        masks: str = "mask_probs",
        kept_scores_as: str = "kept_scores",
        kept_classes_as: str = "kept_classes",
        kept_masks_as: str = "kept_masks",
    ):
        self.score_threshold = score_threshold
        self.scores = scores
        self.classes = classes
        self.masks = masks
        self.kept_scores_as = kept_scores_as
        self.kept_classes_as = kept_classes_as
        self.kept_masks_as = kept_masks_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        query_scores = registry[self.scores]
        keep = query_scores >= self.score_threshold
        registry[self.kept_scores_as] = query_scores[keep].astype(np.float32, copy=False)
        registry[self.kept_classes_as] = registry[self.classes][keep].astype(np.int64, copy=False)
        registry[self.kept_masks_as] = registry[self.masks][keep].astype(np.float32, copy=False)
        return registry


class PanopticWinnerIds:
    def __init__(
        self,
        scores: str = "kept_scores",
        masks: str = "kept_masks",
        as_: str = "winner_ids",
    ):
        self.scores = scores
        self.masks = masks
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        kept_scores = registry[self.scores]
        kept_masks = registry[self.masks]
        if kept_scores.size == 0:
            height, width = kept_masks.shape[-2:]
            registry[self.as_] = np.zeros((height, width), dtype=np.int64)
            return registry
        weighted_masks = kept_scores[:, None, None] * kept_masks
        registry[self.as_] = weighted_masks.argmax(axis=0).astype(np.int64, copy=False)
        return registry


class PanopticSegmentsFromQueries:
    def __init__(
        self,
        thing_class_ids: frozenset[int],
        scores: str = "kept_scores",
        classes: str = "kept_classes",
        masks: str = "kept_masks",
        winner_ids: str = "winner_ids",
        mask_threshold: float = 0.5,
        overlap_threshold: float = 0.8,
    ):
        self.thing_class_ids = thing_class_ids
        self.scores = scores
        self.classes = classes
        self.masks = masks
        self.winner_ids = winner_ids
        self.mask_threshold = mask_threshold
        self.overlap_threshold = overlap_threshold

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        kept_scores = registry[self.scores]
        kept_classes = registry[self.classes]
        kept_masks = registry[self.masks]
        image_shape = tuple(int(dim) for dim in kept_masks.shape[-2:])

        if kept_scores.size == 0:
            masks, scores, classes = _empty_numpy_segments(image_shape)
            registry["masks"] = masks
            registry["scores"] = scores
            registry["classes"] = classes
            return registry

        winner_ids = registry[self.winner_ids]
        merged_masks: list[np.ndarray] = []
        merged_scores: list[float] = []
        merged_classes: list[int] = []
        stuff_lookup: dict[int, int] = {}

        for index in range(kept_scores.shape[0]):
            class_id = int(kept_classes[index])
            query_mask = kept_masks[index] >= self.mask_threshold
            winner_mask = winner_ids == index
            original_area = int(query_mask.sum())
            winner_area = int(winner_mask.sum())
            final_mask = query_mask & winner_mask
            final_area = int(final_mask.sum())
            if original_area == 0 or winner_area == 0 or final_area == 0:
                continue
            if winner_area / original_area < self.overlap_threshold:
                continue
            _append_or_merge_numpy_panoptic_segment(
                merged_masks=merged_masks,
                merged_scores=merged_scores,
                merged_classes=merged_classes,
                stuff_lookup=stuff_lookup,
                class_id=class_id,
                score=float(kept_scores[index]),
                mask=final_mask,
                thing_class_ids=self.thing_class_ids,
            )

        masks, scores, classes = _finalize_numpy_segments(
            merged_masks=merged_masks,
            merged_scores=merged_scores,
            merged_classes=merged_classes,
            image_shape=image_shape,
        )
        registry["masks"] = masks
        registry["scores"] = scores
        registry["classes"] = classes
        return registry


class InstanceTopKPredictions:
    def __init__(
        self,
        class_probs: str = "class_probs",
        mask_probs: str = "mask_probs",
        top_k: int = 100,
        top_scores_as: str = "top_scores",
        class_ids_as: str = "class_ids",
        selected_masks_as: str = "selected_masks",
    ):
        self.class_probs = class_probs
        self.mask_probs = mask_probs
        self.top_k = top_k
        self.top_scores_as = top_scores_as
        self.class_ids_as = class_ids_as
        self.selected_masks_as = selected_masks_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        class_probs = registry[self.class_probs]
        mask_probs = registry[self.mask_probs]
        num_queries, num_classes = class_probs.shape
        if num_queries == 0 or num_classes == 0:
            registry[self.top_scores_as] = np.zeros((0,), dtype=np.float32)
            registry[self.class_ids_as] = np.zeros((0,), dtype=np.int64)
            registry[self.selected_masks_as] = mask_probs[:0].astype(np.float32, copy=False)
            return registry

        flat_scores = class_probs.reshape(-1)
        top_k = min(self.top_k, int(flat_scores.size))
        top_indices = np.argpartition(flat_scores, -top_k)[-top_k:]
        order = np.argsort(flat_scores[top_indices])[::-1]
        top_indices = top_indices[order]
        registry[self.top_scores_as] = flat_scores[top_indices].astype(np.float32, copy=False)
        query_indices = top_indices // num_classes
        registry[self.class_ids_as] = (top_indices % num_classes).astype(np.int64, copy=False)
        registry[self.selected_masks_as] = mask_probs[query_indices].astype(np.float32, copy=False)
        return registry


class InstanceScoreMasks:
    def __init__(
        self,
        mask_threshold: float = 0.5,
        top_scores: str = "top_scores",
        selected_masks: str = "selected_masks",
        binary_masks_as: str = "binary_masks",
        final_scores_as: str = "final_scores",
    ):
        self.mask_threshold = mask_threshold
        self.top_scores = top_scores
        self.selected_masks = selected_masks
        self.binary_masks_as = binary_masks_as
        self.final_scores_as = final_scores_as

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        top_scores = registry[self.top_scores]
        selected_masks = registry[self.selected_masks]
        binary_masks = selected_masks >= self.mask_threshold
        areas = binary_masks.reshape(binary_masks.shape[0], -1).sum(axis=1)
        mask_sums = (selected_masks * binary_masks).reshape(selected_masks.shape[0], -1).sum(axis=1)
        mean_mask_scores = np.where(areas > 0, mask_sums / np.clip(areas, 1, None), 0.0).astype(np.float32, copy=False)
        registry[self.binary_masks_as] = binary_masks
        registry[self.final_scores_as] = (top_scores * mean_mask_scores).astype(np.float32, copy=False)
        return registry


class InstanceSegmentsFromPredictions:
    def __init__(
        self,
        score_threshold: float = 0.5,
        binary_masks: str = "binary_masks",
        final_scores: str = "final_scores",
        class_ids: str = "class_ids",
    ):
        self.score_threshold = score_threshold
        self.binary_masks = binary_masks
        self.final_scores = final_scores
        self.class_ids = class_ids

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        binary_masks = registry[self.binary_masks]
        final_scores = registry[self.final_scores]
        class_ids = registry[self.class_ids]
        image_shape = tuple(int(dim) for dim in binary_masks.shape[-2:])
        areas = binary_masks.reshape(binary_masks.shape[0], -1).sum(axis=1)
        keep = (areas > 0) & (final_scores >= self.score_threshold)

        if not np.any(keep):
            masks, scores, classes = _empty_numpy_segments(image_shape)
        else:
            masks = binary_masks[keep].astype(bool, copy=False)
            scores = final_scores[keep].astype(np.float32, copy=False)
            classes = class_ids[keep]
            order = np.argsort(scores)[::-1]
            masks = masks[order]
            scores = scores[order]
            classes = classes[order]

        registry["masks"] = masks
        registry["scores"] = scores
        registry["classes"] = classes
        return registry


class MasksToBoxes:
    def __init__(self, masks: str = "masks", as_: str = "boxes"):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Mask2Former example where inference stays in Torch and "
            "post-processing crosses into NumPy immediately after raw outputs."
        )
    )
    add_assets_dir_arg(parser)
    parser.add_argument("--output", type=Path, default=None, help="Optional output path prefix for annotated images.")
    parser.add_argument("--task", choices=("panoptic", "instance", "both"), default="both")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k query/class pairs kept for instance mode.")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.8)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for model execution before the NumPy handoff.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.assets_dir / COCO_IMAGE_NAME
    download_if_missing(COCO_IMAGE_URL, image_path)

    for task in resolve_task_list(args.task):
        bundle = LoadedMask2Former.load(task=task, device=args.device)
        output_path = resolve_output_path(args.output, args.assets_dir, task, "numpy")
        record_fields = {
            "index": lambda p: list(range(len(p.classes))),
            "class_id": "classes",
            "class_name": lambda p, names=bundle.class_names: [
                names[int(class_id)] if 0 <= int(class_id) < len(names) else str(class_id) for class_id in p.classes
            ],
            "score": "scores",
            "area": lambda p: [int(np.asarray(mask, dtype=bool).sum()) for mask in p.masks],
            "box": "boxes",
        }

        if task == "panoptic":
            pipeline = Pipeline([
                Mask2FormerInfer(bundle=bundle, device=args.device),
                ToNumpyRegistry(),
                Recall("image_shape"),
                ResizeMasksToStoredImage(),
                ClassProbabilities(),
                MaskProbabilities(),
                PanopticQueryPredictions(),
                PanopticKeepQueries(score_threshold=args.score_threshold),
                PanopticWinnerIds(),
                PanopticSegmentsFromQueries(
                    thing_class_ids=bundle.thing_class_ids,
                    mask_threshold=args.mask_threshold,
                    overlap_threshold=args.overlap_threshold,
                ),
                MasksToBoxes(),
                ToSegmentations(),
                inline(visualize_and_store(output_path, bundle.class_names)),
                MapToObjects(fields=record_fields, at=1),
                LogDetections(bundle.model_id, image_path, output_path, at=1),
            ])
        else:
            pipeline = Pipeline([
                Mask2FormerInfer(bundle=bundle, device=args.device),
                ToNumpyRegistry(),
                Recall("image_shape"),
                ResizeMasksToStoredImage(),
                ClassProbabilities(),
                MaskProbabilities(),
                InstanceTopKPredictions(top_k=args.top_k),
                InstanceScoreMasks(mask_threshold=args.mask_threshold),
                InstanceSegmentsFromPredictions(score_threshold=args.score_threshold),
                MasksToBoxes(),
                ToSegmentations(),
                inline(visualize_and_store(output_path, bundle.class_names)),
                MapToObjects(fields=record_fields, at=1),
                LogDetections(bundle.model_id, image_path, output_path, at=1),
            ])

        pipeline = build_mask2former_preprocess_pipeline() + pipeline
        pipeline.validate()
        pipeline(image_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
