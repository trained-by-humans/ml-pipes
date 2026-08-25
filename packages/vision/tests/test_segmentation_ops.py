from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import (
    DrawMasks,
    FilterTensorsByMasksArea,
    ImagePayload,
    MasksToBoxes,
    MeanMaskScores,
    ProjectMasks,
    ProjectRoIMasks,
    ReconstructMasks,
    ResizeMasks,
    ResizeTransform,
    WeightMasksByScores,
)


def _make_registry(boxes, scores, classes):
    registry = TensorRegistry()
    registry["boxes"] = np.array(boxes, dtype=np.float32)
    registry["scores"] = np.array(scores, dtype=np.float32)
    registry["classes"] = np.array(classes, dtype=np.int32)
    return registry


def test_weight_masks_by_scores_broadcasts_scores_over_masks():
    registry = TensorRegistry(
        {
            "scores": np.array([0.5, 2.0], dtype=np.float32),
            "masks": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            ),
        }
    )

    result = WeightMasksByScores(as_="weighted_masks")(registry)

    assert np.allclose(
        result["weighted_masks"],
        [
            [[0.5, 1.0], [1.5, 2.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ],
    )


def test_resize_masks_to_image_resizes_mask_stack():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                ],
                dtype=np.float32,
            )
        }
    )

    result = ResizeMasks(masks="masks", as_="resized_masks")(registry, (4, 6))

    assert result["resized_masks"].shape == (1, 4, 6)
    assert result["resized_masks"].dtype == np.float32


def test_mean_mask_scores_computes_mean_over_mask_support():
    registry = TensorRegistry(
        {
            "selected_masks": np.array(
                [
                    [[0.0, 1.0], [0.5, 0.0]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ],
                dtype=np.float32,
            ),
            "masks": np.array(
                [
                    [[False, True], [True, False]],
                    [[True, False], [False, True]],
                ]
            ),
        }
    )

    result = MeanMaskScores(mask_scores="selected_masks", as_="mean_mask_scores")(registry)

    assert np.allclose(result["mean_mask_scores"], [0.75, 0.5])


def test_mean_mask_scores_handles_empty_masks():
    registry = TensorRegistry(
        {
            "selected_masks": np.zeros((0, 2, 2), dtype=np.float32),
            "masks": np.zeros((0, 2, 2), dtype=bool),
        }
    )

    result = MeanMaskScores(mask_scores="selected_masks", as_="mean_mask_scores")(registry)

    assert result["mean_mask_scores"].shape == (0,)
    assert result["mean_mask_scores"].dtype == np.float64


def test_masks_to_boxes_converts_masks_to_xyxy():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[False, True, True], [False, True, False]],
                    [[False, False, False], [False, False, False]],
                ]
            )
        }
    )

    result = MasksToBoxes(as_="boxes")(registry)

    assert np.allclose(result["boxes"][0], [1.0, 0.0, 3.0, 2.0])
    assert np.allclose(result["boxes"][1], [0.0, 0.0, 0.0, 0.0])


def test_filter_masks_by_area_filters_parallel_tensors():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=bool,
            ),
            "scores": np.array([0.2, 0.9], dtype=np.float32),
            "classes": np.array([1, 2], dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert result["masks"].shape[0] == 1
    assert np.allclose(result["scores"], [0.9])
    assert result["classes"].tolist() == [2]


def test_filter_masks_by_area_handles_empty_masks():
    registry = TensorRegistry(
        {
            "masks": np.zeros((0, 2, 2), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert result["masks"].shape == (0, 2, 2)
    assert result["scores"].shape == (0,)
    assert result["classes"].shape == (0,)


def test_filter_tensors_by_masks_area_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=bool,
            ),
            "scores": np.array([0.2, 0.9], dtype=np.float32),
            "classes": np.array([1, 2], dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea(
        "scores",
        "classes",
        masks="masks",
        min_area=2,
        as_=("selected_masks", "selected_scores", "selected_classes"),
    )(registry)

    assert result["selected_masks"].shape[0] == 1
    assert np.allclose(result["selected_scores"], [0.9])
    assert result["selected_classes"].tolist() == [2]
    assert np.allclose(result["scores"], [0.2, 0.9])


def test_reconstruct_masks_produces_correct_shape():
    coefficients = np.ones((2, 3), dtype=np.float32)
    prototypes = np.ones((3, 4, 4), dtype=np.float32)
    registry = TensorRegistry({"coefficients": coefficients, "prototypes": prototypes})

    result = ReconstructMasks("coefficients", "prototypes", as_="masks")(registry)

    assert result["masks"].shape == (2, 4, 4)


def test_project_masks_produces_binary_masks_at_original_size():
    coefficients = np.array([[1.0]], dtype=np.float32)
    prototypes = np.ones((1, 4, 4), dtype=np.float32)
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(0.0, 0.0),
        original_shape=(2, 2),
        resized_shape=(4, 4),
    )
    registry = TensorRegistry()
    registry["boxes"] = np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    registry["scores"] = np.array([0.9], dtype=np.float32)
    registry["classes"] = np.array([1], dtype=np.int32)
    registry["masks"] = (coefficients @ prototypes.reshape(1, -1)).reshape(1, 4, 4)

    result = ProjectMasks(mask_threshold=0.0)(registry, transform)

    assert isinstance(result["masks"], np.ndarray)
    assert result["masks"].shape == (1, 2, 2)
    assert result["masks"][0].shape == (2, 2)


def test_project_masks_returns_empty_mask_array_for_empty_input():
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(0.0, 0.0),
        original_shape=(2, 3),
        resized_shape=(4, 6),
    )
    registry = TensorRegistry()
    registry["boxes"] = np.zeros((0, 4), dtype=np.float32)
    registry["masks"] = np.zeros((0, 4, 4), dtype=np.float32)

    result = ProjectMasks(mask_threshold=0.0)(registry, transform)

    assert isinstance(result["masks"], np.ndarray)
    assert result["masks"].shape == (0, 2, 3)
    assert result["masks"].dtype == np.uint8


def test_draw_masks_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ImagePayload(array=image, color_space="BGR", layout="HWC")
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    registry = _make_registry(boxes=[[8.0, 8.0, 24.0, 24.0]], scores=[0.9], classes=[1])
    registry["masks"] = np.asarray([mask])

    result, returned_registry = DrawMasks(alpha=0.6)(source, registry)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_draw_masks_accepts_class_agnostic_masks():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    registry = TensorRegistry({"masks": np.asarray([mask])})

    result, returned_registry = DrawMasks(classes=None)(
        ImagePayload(array=np.zeros((4, 4, 3), dtype=np.uint8), color_space="BGR", layout="HWC"),
        registry,
    )

    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_draw_masks_requires_classes_when_class_names_are_configured():
    with pytest.raises(ValueError, match="class_names requires a classes tensor"):
        DrawMasks(classes=None, class_names=["person"])


def test_draw_masks_translates_bgr_palette_for_rgb_images():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    registry = _make_registry(boxes=[[0.0, 0.0, 1.0, 1.0]], scores=[0.9], classes=[0])
    registry["masks"] = np.asarray([[[True]]])

    result, _ = DrawMasks(alpha=1.0)(ImagePayload(array=image, color_space="RGB", layout="HWC"), registry)

    assert result.color_space == "RGB"
    assert result.array[0, 0].tolist() == [191, 53, 17]


def test_draw_masks_accepts_source_names_before_rendering_options():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    registry = _make_registry(boxes=[[8.0, 8.0, 24.0, 24.0]], scores=[0.9], classes=[1])
    registry["instance_masks"] = np.asarray([mask])
    registry["labels"] = np.asarray([1])

    result, returned_registry = DrawMasks("instance_masks", "labels", alpha=0.6)(
        ImagePayload(array=image, color_space="BGR", layout="HWC"), registry
    )

    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_project_roi_masks_embeds_mask_into_canvas():
    transform = ResizeTransform(scale=(1.0, 1.0), pad=(0.0, 0.0), original_shape=(8, 8), resized_shape=(8, 8))
    registry = TensorRegistry()
    registry["boxes"] = np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32)
    registry["masks"] = np.ones((1, 4, 4), dtype=np.float32)

    result = ProjectRoIMasks(mask_threshold=0.5)(registry, transform)

    canvas = result["masks"]
    assert canvas.shape == (1, 8, 8)
    assert np.all(canvas[0, 2:6, 2:6])
    assert not np.any(canvas[0, :2, :])


def test_project_roi_masks_preserves_fractional_box():
    transform = ResizeTransform(scale=(1.0, 1.0), pad=(0.0, 0.0), original_shape=(4, 4), resized_shape=(4, 4))
    registry = TensorRegistry()
    registry["boxes"] = np.array([[1.1, 1.1, 1.9, 1.9]], dtype=np.float32)
    registry["masks"] = np.ones((1, 1, 1), dtype=np.float32)

    result = ProjectRoIMasks(mask_threshold=0.5)(registry, transform)

    assert np.any(result["masks"][0]), "mask was silently dropped due to truncation"
