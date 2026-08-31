"""
Mask2Former segmentation with Torch-side postprocess.

Requires `torch`, `transformers`, `safetensors`, and `scipy`.

Run from the repo root:
    python examples/torch/run_mask2former_torch_postprocess.py
    python examples/torch/run_mask2former_torch_postprocess.py --task panoptic --output mask2former.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.torch"

from examples.common import ASSETS_DIR, COCO_IMAGE_NAME, COCO_IMAGE_URL, resolve_input_path, visualize_and_store
from ml_pipes.core import (
    Pipeline,
    inline,
)
from ml_pipes.standard import Recall
from ml_pipes.vision import LogDetections
from ml_pipes.torch import (
    ToNumpyRegistry,
    ArgMax,
    BinarizeTensorByThreshold,
    TorchFilterTensorsByMasksArea,
    TorchFilterTensorsByScore,
    GatherScores,
    TorchMasksToBoxes,
    TorchMeanMaskScores,
    MultiplyTensors,
    TorchResizeMasks,
    SelectTensors,
    Sigmoid,
    Slice,
    SortTensorsBy,
    Softmax,
    TopKIndices2D,
    TorchWeightMasksByScores,
)
from ml_pipes.torch.types import TensorRegistry
from .mask2former_infer import (
    Mask2FormerBundle,
    add_mask2former_args,
    build_mask2former_infer_pipeline,
    resolve_output_path,
)

_TOP_K = 100
_SCORE_THRESHOLD = 0.5
_MASK_THRESHOLD = 0.5
_OVERLAP_THRESHOLD = 0.8


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

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
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
            final_mask = query_mask & winner_mask
            final_area = int(final_mask.sum().item())
            if original_area == 0 or final_area == 0:
                continue
            if final_area / original_area < self.overlap_threshold:
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


def build_torch_postprocess_pipeline(
    bundle: Mask2FormerBundle,
    input_path: Path,
    output_path: Path,
) -> Pipeline[TensorRegistry, object]:
    if bundle.task == "panoptic":
        postprocess_pipeline = Pipeline([
            Recall("image_shape"),
            TorchResizeMasks(masks="masks_queries_logits"),
            Softmax("class_queries_logits", as_="class_probs"),
            Slice("class_probs", slice(None, -1)),
            Sigmoid("masks_queries_logits", as_="mask_probs"),
            ArgMax("class_probs", as_="query_classes"),
            GatherScores("class_probs", "query_classes", as_="query_scores"),
            TorchFilterTensorsByScore(
                "query_classes", "mask_probs", score="query_scores", min_score=_SCORE_THRESHOLD
            ),
            TorchWeightMasksByScores("mask_probs", "query_scores", as_="weighted_masks"),
            ArgMax("weighted_masks", axis=0, as_="winner_ids"),
            TorchPanopticSegmentsFromQueries(
                thing_class_ids=bundle.thing_class_ids,
                scores="query_scores",
                classes="query_classes",
                masks="mask_probs",
                winner_ids="winner_ids",
                mask_threshold=_MASK_THRESHOLD,
                overlap_threshold=_OVERLAP_THRESHOLD,
            ),
            TorchMasksToBoxes(as_="boxes"),
            ToNumpyRegistry(),
            inline(visualize_and_store(output_path, bundle.class_names)),
            LogDetections(bundle.model_id, input_path, output_path, at=1),
        ])
    else:
        postprocess_pipeline = Pipeline([
            Recall("image_shape"),
            TorchResizeMasks(masks="masks_queries_logits"),
            Softmax("class_queries_logits", as_="class_probs"),
            Slice("class_probs", slice(None, -1)),
            Sigmoid("masks_queries_logits", as_="mask_probs"),
            TopKIndices2D(
                "class_probs",
                k=_TOP_K,
                values_as="top_scores",
                row_indices_as="query_indices",
                col_indices_as="classes",
            ),
            SelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
            BinarizeTensorByThreshold("selected_masks", threshold=_MASK_THRESHOLD, as_="masks"),
            TorchMeanMaskScores(mask_scores="selected_masks", masks="masks", as_="mean_mask_scores"),
            MultiplyTensors("top_scores", "mean_mask_scores", as_="scores"),
            TorchFilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=1),
            TorchFilterTensorsByScore("masks", "classes", score="scores", min_score=_SCORE_THRESHOLD),
            SortTensorsBy("masks", "classes", by="scores"),
            TorchMasksToBoxes(masks="masks", as_="boxes"),
            ToNumpyRegistry(),
            inline(visualize_and_store(output_path, bundle.class_names)),
            LogDetections(bundle.model_id, input_path, output_path, at=1),
        ])

    return postprocess_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Mask2Former example where inference and post-processing stay in Torch "
            "until the final conversion back to NumPy segmentations."
        )
    )
    add_mask2former_args(parser, device_help="Torch device for model execution and Torch postprocess.")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = resolve_output_path(args.output, input_path.name, args.task, "torch")

    bundle = Mask2FormerBundle.load(task=args.task, device=args.device)
    infer_pipeline = build_mask2former_infer_pipeline(bundle, args.device)
    postprocess_pipeline = build_torch_postprocess_pipeline(bundle, input_path, output_path)

    pipeline = infer_pipeline + postprocess_pipeline
    pipeline.validate()
    pipeline(input_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
