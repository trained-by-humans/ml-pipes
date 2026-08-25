from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml_pipes.torch import (
    TorchConvertBoxFormat,
    TorchFilterTensorsByClasses,
    TorchFilterTensorsByMasksArea,
    TorchFilterTensorsByScore,
    TorchMasksToBoxes,
    TorchMeanMaskedScores,
    TorchNMS,
    TorchReconstructMasks,
    TorchResizeMasks,
    TorchWeightMasksByScores,
)
from ml_pipes.torch.types import TorchTensorRegistry


def test_torch_convert_box_format_cxcywh_to_xyxy():
    registry = TorchTensorRegistry({"boxes": torch.tensor([[10.0, 20.0, 4.0, 6.0]], dtype=torch.float32)})

    result = TorchConvertBoxFormat(from_="cxcywh")(registry)

    assert torch.allclose(result["boxes"], torch.tensor([[8.0, 17.0, 12.0, 23.0]], dtype=torch.float32))


def test_torch_weight_masks_by_scores_broadcasts_scores_over_masks():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.5, 2.0], dtype=torch.float32),
            "masks": torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=torch.float32,
            ),
        }
    )

    result = TorchWeightMasksByScores(as_="weighted_masks")(registry)

    assert torch.allclose(
        result["weighted_masks"],
        torch.tensor(
            [
                [[0.5, 1.0], [1.5, 2.0]],
                [[10.0, 12.0], [14.0, 16.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_torch_resize_masks_to_image_resizes_mask_stack():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
    )

    TorchResizeMasks(masks="masks", as_="resized_masks")(registry, (4, 6))

    assert registry["resized_masks"].shape == (1, 4, 6)
    assert registry["resized_masks"].dtype == torch.float32


def test_torch_mean_masked_scores_computes_mean_over_mask_support():
    registry = TorchTensorRegistry(
        {
            "selected_masks": torch.tensor(
                [
                    [[0.0, 1.0], [0.5, 0.0]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ],
                dtype=torch.float32,
            ),
            "masks": torch.tensor(
                [
                    [[False, True], [True, False]],
                    [[True, False], [False, True]],
                ],
                dtype=torch.bool,
            ),
        }
    )

    TorchMeanMaskedScores(mask_scores="selected_masks", as_="mean_mask_scores")(registry)

    assert torch.allclose(registry["mean_mask_scores"], torch.tensor([0.75, 0.5]))


def test_torch_mean_masked_scores_handles_empty_masks():
    registry = TorchTensorRegistry(
        {
            "selected_masks": torch.zeros((0, 2, 2), dtype=torch.float32),
            "masks": torch.zeros((0, 2, 2), dtype=torch.bool),
        }
    )

    result = TorchMeanMaskedScores(mask_scores="selected_masks", as_="mean_mask_scores")(registry)

    assert tuple(result["mean_mask_scores"].shape) == (0,)
    assert result["mean_mask_scores"].dtype == torch.float32


def test_torch_masks_to_boxes_converts_masks_to_xyxy():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[False, True, True], [False, True, False]],
                    [[False, False, False], [False, False, False]],
                ],
                dtype=torch.bool,
            )
        }
    )

    TorchMasksToBoxes(as_="boxes")(registry)

    assert torch.allclose(registry["boxes"][0], torch.tensor([1.0, 0.0, 3.0, 2.0]))
    assert torch.allclose(registry["boxes"][1], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_torch_filter_tensors_by_score_filters_parallel_tensors():
    registry = TorchTensorRegistry(
        {
            "query_scores": torch.tensor([0.9, 0.1], dtype=torch.float32),
            "query_classes": torch.tensor([1, 0], dtype=torch.int64),
            "mask_probs": torch.tensor(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ],
                dtype=torch.float32,
            ),
        }
    )

    TorchFilterTensorsByScore("query_classes", "mask_probs", score="query_scores", min_score=0.2)(registry)

    assert registry["query_scores"].shape == (1,)
    assert registry["query_classes"].tolist() == [1]
    assert registry["mask_probs"].shape == (1, 2, 2)


def test_torch_filter_tensors_by_masks_area_filters_parallel_tensors():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=torch.bool,
            ),
            "scores": torch.tensor([0.5, 0.9], dtype=torch.float32),
            "classes": torch.tensor([5, 9], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert registry["masks"].shape[0] == 1
    assert torch.allclose(registry["scores"], torch.tensor([0.9]))
    assert registry["classes"].tolist() == [9]


def test_torch_filter_tensors_by_masks_area_handles_empty_masks():
    registry = TorchTensorRegistry(
        {
            "masks": torch.zeros((0, 2, 2), dtype=torch.bool),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "classes": torch.zeros((0,), dtype=torch.int64),
        }
    )

    result = TorchFilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert tuple(result["masks"].shape) == (0, 2, 2)
    assert tuple(result["scores"].shape) == (0,)
    assert tuple(result["classes"].shape) == (0,)


def test_torch_filter_tensors_by_score_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByScore(
        "classes",
        score="scores",
        min_score=0.75,
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9, 0.8]))
    assert registry["selected_classes"].tolist() == [0, 0]
    assert torch.allclose(registry["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_filter_tensors_by_masks_area_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=torch.bool,
            ),
            "scores": torch.tensor([0.2, 0.9], dtype=torch.float32),
            "classes": torch.tensor([1, 2], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByMasksArea(
        "scores",
        "classes",
        masks="masks",
        min_area=2,
        as_=("selected_masks", "selected_scores", "selected_classes"),
    )(registry)

    assert registry["selected_masks"].shape[0] == 1
    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9]))
    assert registry["selected_classes"].tolist() == [2]
    assert torch.allclose(registry["scores"], torch.tensor([0.2, 0.9]))


def test_torch_filter_tensors_by_classes_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 2], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByClasses(
        "scores",
        classes="classes",
        keep_classes=[0, 2],
        as_=("selected_classes", "selected_scores"),
    )(registry)

    assert registry["selected_classes"].tolist() == [0, 2]
    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9, 0.8]))
    assert registry["classes"].tolist() == [0, 1, 2]


def test_torch_reconstruct_masks_produces_correct_shape():
    coefficients = torch.ones((2, 3), dtype=torch.float32)
    prototypes = torch.ones((3, 4, 4), dtype=torch.float32)
    registry = TorchTensorRegistry({"coefficients": coefficients, "prototypes": prototypes})

    result = TorchReconstructMasks("coefficients", "prototypes", as_="masks")(registry)

    assert result["masks"].shape == (2, 4, 4)


def test_torch_nms_keeps_overlapping_boxes_from_different_classes():
    pytest.importorskip("torchvision")
    registry = TorchTensorRegistry(
        {
            "boxes": torch.tensor([[10, 10, 50, 50], [12, 12, 48, 48]], dtype=torch.float32),
            "scores": torch.tensor([0.95, 0.9], dtype=torch.float32),
            "classes": torch.tensor([0, 1], dtype=torch.int64),
        }
    )

    result = TorchNMS()(registry)

    assert tuple(result["boxes"].shape) == (2, 4)
    assert result["classes"].tolist() == [0, 1]


def test_torch_nms_suppresses_same_class_overlap():
    pytest.importorskip("torchvision")
    registry = TorchTensorRegistry(
        {
            "boxes": torch.tensor([[10, 10, 50, 50], [12, 12, 48, 48], [100, 100, 140, 140]], dtype=torch.float32),
            "scores": torch.tensor([0.95, 0.85, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 0, 0], dtype=torch.int64),
        }
    )

    result = TorchNMS()(registry)

    assert tuple(result["boxes"].shape) == (2, 4)
    assert torch.allclose(result["scores"], torch.tensor([0.95, 0.8]))


def test_torch_nms_filters_and_stores_indices():
    pytest.importorskip("torchvision")
    registry = TorchTensorRegistry(
        {
            "boxes": torch.tensor(
                [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32),
            "classes": torch.tensor([0, 0, 1], dtype=torch.int64),
        }
    )

    result = TorchNMS(kept_as="kept", iou_threshold=0.5)(registry)

    assert result["boxes"].shape[0] == 2
    assert result["kept"].dtype == torch.int64
    assert result["kept"].tolist() == [2, 0]
