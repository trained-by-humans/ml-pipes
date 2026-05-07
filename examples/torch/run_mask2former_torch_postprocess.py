from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.torch"

from examples.common import COCO_IMAGE_NAME, COCO_IMAGE_URL, add_assets_dir_arg, download_if_missing, visualize_and_store
from ml_pipes import LogDetections, MapToObjects, Pipeline, Recall, ToSegmentations, inline
from ml_pipes.torch import (
    ToNumpyRegistry,
    TorchArgMax,
    TorchGatherScores,
    TorchSigmoid,
    TorchSlice,
    TorchSoftmax,
    TorchThresholdTensors,
    TorchWeightMasksByScores,
)
from ml_pipes.torch.types import TorchTensorRegistry
from .mask2former_infer import (
    Mask2FormerInfer,
    LoadedMask2Former,
    build_mask2former_preprocess_pipeline,
    resolve_output_path,
    resolve_task_list,
)


class TorchResizeMasksToImage:
    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TorchTensorRegistry, image_shape: tuple[int, int]) -> TorchTensorRegistry:
        masks = registry[self.src]
        resized = F.interpolate(masks[:, None, :, :], size=image_shape, mode="bilinear", align_corners=False)[:, 0]
        registry[self.as_] = resized
        return registry


def _empty_torch_segments(
    device: torch.device | str,
    image_shape: tuple[int, int],
    score_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = image_shape
    return (
        torch.zeros((0, height, width), dtype=torch.bool, device=device),
        torch.zeros((0,), dtype=score_dtype, device=device),
        torch.zeros((0,), dtype=torch.int64, device=device),
    )


def _append_or_merge_torch_panoptic_segment(
    merged_masks: list[torch.Tensor],
    merged_scores: list[torch.Tensor],
    merged_classes: list[int],
    stuff_lookup: dict[int, int],
    class_id: int,
    score: torch.Tensor,
    mask: torch.Tensor,
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
    merged_scores[existing] = torch.maximum(merged_scores[existing], score)


def _finalize_torch_segments(
    merged_masks: list[torch.Tensor],
    merged_scores: list[torch.Tensor],
    merged_classes: list[int],
    device: torch.device,
    image_shape: tuple[int, int],
    score_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not merged_masks:
        return _empty_torch_segments(device, image_shape, score_dtype=score_dtype)
    return (
        torch.stack(merged_masks, dim=0).to(dtype=torch.bool),
        torch.stack(merged_scores, dim=0),
        torch.tensor(merged_classes, dtype=torch.int64, device=device),
    )


class TorchPanopticSegmentsFromQueries:
    def __init__(
        self,
        thing_class_ids: frozenset[int],
        scores: str = "query_scores",
        classes: str = "query_classes",
        masks: str = "mask_probs",
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        kept_scores = registry[self.scores]
        kept_classes = registry[self.classes]
        kept_masks = registry[self.masks]
        image_shape = tuple(int(dim) for dim in kept_masks.shape[-2:])

        if kept_scores.numel() == 0:
            masks, scores, classes = _empty_torch_segments(kept_masks.device, image_shape, score_dtype=kept_scores.dtype)
            registry["masks"] = masks
            registry["scores"] = scores
            registry["classes"] = classes
            return registry

        winner_ids = registry[self.winner_ids]
        merged_masks: list[torch.Tensor] = []
        merged_scores: list[torch.Tensor] = []
        merged_classes: list[int] = []
        stuff_lookup: dict[int, int] = {}

        for index in range(kept_scores.shape[0]):
            class_id = int(kept_classes[index].item())
            query_mask = kept_masks[index] >= self.mask_threshold
            winner_mask = winner_ids == index
            original_area = int(query_mask.sum().item())
            winner_area = int(winner_mask.sum().item())
            final_mask = query_mask & winner_mask
            final_area = int(final_mask.sum().item())
            if original_area == 0 or winner_area == 0 or final_area == 0:
                continue
            if winner_area / original_area < self.overlap_threshold:
                continue
            _append_or_merge_torch_panoptic_segment(
                merged_masks=merged_masks,
                merged_scores=merged_scores,
                merged_classes=merged_classes,
                stuff_lookup=stuff_lookup,
                class_id=class_id,
                score=kept_scores[index],
                mask=final_mask,
                thing_class_ids=self.thing_class_ids,
            )

        masks, scores, classes = _finalize_torch_segments(
            merged_masks=merged_masks,
            merged_scores=merged_scores,
            merged_classes=merged_classes,
            device=kept_masks.device,
            image_shape=image_shape,
            score_dtype=kept_scores.dtype,
        )
        registry["masks"] = masks
        registry["scores"] = scores
        registry["classes"] = classes
        return registry


class TorchInstanceTopKPredictions:
    def __init__(
        self,
        class_logits: str,
        mask_logits: str,
        top_k: int = 100,
        top_scores_as: str = "top_scores",
        class_ids_as: str = "class_ids",
        selected_masks_as: str = "selected_masks",
    ):
        self.class_logits = class_logits
        self.mask_logits = mask_logits
        self.top_k = top_k
        self.top_scores_as = top_scores_as
        self.class_ids_as = class_ids_as
        self.selected_masks_as = selected_masks_as

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        class_probs = registry[self.class_logits].softmax(dim=-1)[..., :-1]
        mask_probs = torch.sigmoid(registry[self.mask_logits])
        num_queries, num_classes = class_probs.shape
        if num_queries == 0 or num_classes == 0:
            registry[self.top_scores_as] = torch.zeros((0,), dtype=class_probs.dtype, device=mask_probs.device)
            registry[self.class_ids_as] = torch.zeros((0,), dtype=torch.int64, device=mask_probs.device)
            registry[self.selected_masks_as] = mask_probs[:0]
            return registry

        flat_scores = class_probs.reshape(-1)
        top_k = min(self.top_k, int(flat_scores.numel()))
        top_scores, top_indices = torch.topk(flat_scores, k=top_k)
        query_indices = torch.div(top_indices, num_classes, rounding_mode="floor")
        registry[self.top_scores_as] = top_scores
        registry[self.class_ids_as] = (top_indices % num_classes).to(torch.int64)
        registry[self.selected_masks_as] = mask_probs[query_indices]
        return registry


class TorchInstanceScoreMasks:
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        top_scores = registry[self.top_scores]
        selected_masks = registry[self.selected_masks]
        binary_masks = selected_masks >= self.mask_threshold
        areas = binary_masks.flatten(1).sum(dim=1)
        mean_mask_scores = torch.where(
            areas > 0,
            (selected_masks * binary_masks).flatten(1).sum(dim=1) / areas.clamp_min(1).to(selected_masks.dtype),
            torch.zeros_like(top_scores),
        )
        registry[self.binary_masks_as] = binary_masks
        registry[self.final_scores_as] = top_scores * mean_mask_scores
        return registry


class TorchInstanceSegmentsFromPredictions:
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

    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        binary_masks = registry[self.binary_masks]
        final_scores = registry[self.final_scores]
        class_ids = registry[self.class_ids]
        image_shape = tuple(int(dim) for dim in binary_masks.shape[-2:])
        areas = binary_masks.flatten(1).sum(dim=1)
        keep = (areas > 0) & (final_scores >= self.score_threshold)

        if keep.sum().item() == 0:
            masks, scores, classes = _empty_torch_segments(binary_masks.device, image_shape, score_dtype=final_scores.dtype)
        else:
            masks = binary_masks[keep].to(dtype=torch.bool)
            scores = final_scores[keep]
            classes = class_ids[keep]
            order = torch.argsort(scores, descending=True)
            masks = masks[order]
            scores = scores[order]
            classes = classes[order]

        registry["masks"] = masks
        registry["scores"] = scores
        registry["classes"] = classes
        return registry


class TorchMasksToBoxes:
    def __init__(self, masks: str = "masks", as_: str = "boxes"):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Mask2Former example where inference and post-processing stay in Torch "
            "until the final conversion back to NumPy segmentations."
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
        help="Torch device for model execution and Torch post-processing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.assets_dir / COCO_IMAGE_NAME
    download_if_missing(COCO_IMAGE_URL, image_path)

    for task in resolve_task_list(args.task):
        bundle = LoadedMask2Former.load(task=task, device=args.device)
        output_path = resolve_output_path(args.output, args.assets_dir, task, "torch")
        record_fields = {
            "index": lambda p: list(range(len(p.classes))),
            "class_id": "classes",
            "class_name": lambda p, names=bundle.class_names: [
                names[int(class_id)] if 0 <= int(class_id) < len(names) else str(class_id) for class_id in p.classes
            ],
            "score": "scores",
            "area": lambda p: [int(torch.as_tensor(mask, dtype=torch.bool).sum().item()) for mask in p.masks],
            "box": "boxes",
        }

        if task == "panoptic":
            pipeline = Pipeline([
                Mask2FormerInfer(bundle=bundle, device=args.device),
                Recall("image_shape"),
                TorchResizeMasksToImage(src="masks_queries_logits"),
                TorchSoftmax("class_queries_logits", as_="class_probs"),
                TorchSlice("class_probs", slice(None, -1)),
                TorchSigmoid("masks_queries_logits", as_="mask_probs"),
                TorchArgMax("class_probs", as_="query_classes"),
                TorchGatherScores("class_probs", "query_classes", as_="query_scores"),
                TorchThresholdTensors("query_classes", "mask_probs", score="query_scores", min_score=args.score_threshold),
                TorchWeightMasksByScores("query_scores", "mask_probs"),
                TorchArgMax("weighted_masks", axis=0, as_="winner_ids"),
                TorchPanopticSegmentsFromQueries(
                    thing_class_ids=bundle.thing_class_ids,
                    scores="query_scores",
                    classes="query_classes",
                    masks="mask_probs",
                    winner_ids="winner_ids",
                    mask_threshold=args.mask_threshold,
                    overlap_threshold=args.overlap_threshold,
                ),
                TorchMasksToBoxes(),
                ToNumpyRegistry(),
                ToSegmentations(),
                inline(visualize_and_store(output_path, bundle.class_names)),
                MapToObjects(fields=record_fields, at=1),
                LogDetections(bundle.model_id, image_path, output_path, at=1),
            ])
        else:
            pipeline = Pipeline([
                Mask2FormerInfer(bundle=bundle, device=args.device),
                Recall("image_shape"),
                TorchResizeMasksToImage(src="masks_queries_logits"),
                TorchInstanceTopKPredictions(
                    class_logits="class_queries_logits",
                    mask_logits="masks_queries_logits",
                    top_k=args.top_k,
                ),
                TorchInstanceScoreMasks(mask_threshold=args.mask_threshold),
                TorchInstanceSegmentsFromPredictions(score_threshold=args.score_threshold),
                TorchMasksToBoxes(),
                ToNumpyRegistry(),
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
