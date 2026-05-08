from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.torch"

from examples.common import COCO_IMAGE_NAME, COCO_IMAGE_URL, add_assets_dir_arg, download_if_missing, visualize_and_store
from ml_pipes import LogDetections, MapToObjects, Pipeline, Recall, ToSegmentations, inline
from ml_pipes.torch import (
    ToNumpyRegistry,
    TorchArgMax,
    TorchBinarizeTensor,
    TorchFilterTensorsByMasksArea,
    TorchFilterTensorsByScore,
    TorchGatherScores,
    TorchMasksToBoxes,
    TorchMeanMaskScores,
    TorchMultiplyTensors,
    TorchResizeMasks,
    TorchSelectTensors,
    TorchSigmoid,
    TorchSlice,
    TorchSortTensorsBy,
    TorchSoftmax,
    TorchTopKIndices2D,
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
                TorchResizeMasks(masks="masks_queries_logits"),
                TorchSoftmax("class_queries_logits", as_="class_probs"),
                TorchSlice("class_probs", slice(None, -1)),
                TorchSigmoid("masks_queries_logits", as_="mask_probs"),
                TorchArgMax("class_probs", as_="query_classes"),
                TorchGatherScores("class_probs", "query_classes", as_="query_scores"),
                TorchFilterTensorsByScore(
                    "query_classes", "mask_probs", score="query_scores", min_score=args.score_threshold
                ),
                TorchWeightMasksByScores("mask_probs", "query_scores", as_="weighted_masks"),
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
                TorchMasksToBoxes(as_="boxes"),
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
                TorchResizeMasks(masks="masks_queries_logits"),
                TorchSoftmax("class_queries_logits", as_="class_probs"),
                TorchSlice("class_probs", slice(None, -1)),
                TorchSigmoid("masks_queries_logits", as_="mask_probs"),
                TorchTopKIndices2D(
                    "class_probs",
                    k=args.top_k,
                    values_as="top_scores",
                    row_indices_as="query_indices",
                    col_indices_as="class_ids",
                ),
                TorchSelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
                TorchBinarizeTensor("selected_masks", threshold=args.mask_threshold, as_="binary_masks"),
                TorchMeanMaskScores(masks="selected_masks", as_="mean_mask_scores"),
                TorchMultiplyTensors("top_scores", "mean_mask_scores", as_="final_scores"),
                TorchFilterTensorsByMasksArea("final_scores", "class_ids", masks="binary_masks", min_area=1),
                TorchFilterTensorsByScore("binary_masks", "class_ids", score="final_scores", min_score=args.score_threshold),
                TorchSortTensorsBy("binary_masks", "class_ids", by="final_scores"),
                TorchMasksToBoxes(masks="binary_masks", as_="boxes"),
                ToNumpyRegistry(),
                ToSegmentations(scores="final_scores", classes="class_ids", masks="binary_masks"),
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
