from __future__ import annotations

import numpy as np
import pytest

from ml_pipes import BlendImages, ClampDensity, DensityPrediction, DensityToHeatmap, ImagePayload, SumDensity


def test_clamp_density_zeroes_negative_values_and_coerces_float32() -> None:
    prediction = DensityPrediction(
        density_map=np.array([[-1, 2], [3, -4]], dtype=np.int16),
    )

    clamped = ClampDensity()(prediction)

    assert clamped.density_map.dtype == np.float32
    assert np.array_equal(clamped.density_map, np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32))


def test_clamp_density_preserves_positive_values() -> None:
    original = np.array([[0.25, 1.5], [2.0, 0.0]], dtype=np.float32)

    clamped = ClampDensity()(DensityPrediction(density_map=original))

    assert np.array_equal(clamped.density_map, original)


def test_clamp_density_does_not_mutate_input_array() -> None:
    original = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    prediction = DensityPrediction(density_map=original.copy())

    clamped = ClampDensity()(prediction)

    assert np.array_equal(prediction.density_map, original)
    assert clamped.density_map is not prediction.density_map


def test_sum_density_returns_scalar_sum_of_density_map() -> None:
    prediction = DensityPrediction(
        density_map=np.array([[0.5, 1.5], [2.0, 3.5]], dtype=np.float32),
    )

    counted = SumDensity()(prediction)

    assert counted == 7.5


def test_sum_density_on_empty_map_returns_zero() -> None:
    prediction = DensityPrediction(density_map=np.zeros((0, 0), dtype=np.float32))

    counted = SumDensity()(prediction)

    assert counted == 0.0


def test_density_to_heatmap_matches_requested_size_and_metadata() -> None:
    source = ImagePayload(array=np.zeros((24, 32, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32))

    returned_source, heatmap = DensityToHeatmap()(source, prediction)

    assert returned_source is source
    assert heatmap.array.shape == (24, 32, 3)
    assert heatmap.array.dtype == np.uint8
    assert heatmap.color_space == "BGR"
    assert heatmap.layout == "HWC"


def test_density_to_heatmap_returns_black_image_for_zero_density() -> None:
    source = ImagePayload(array=np.zeros((12, 16, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.zeros((4, 4), dtype=np.float32))

    _, heatmap = DensityToHeatmap()(source, prediction)

    assert np.count_nonzero(heatmap.array) == 0


def test_density_to_heatmap_returns_black_image_for_negative_only_density() -> None:
    source = ImagePayload(array=np.zeros((12, 16, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32))

    _, heatmap = DensityToHeatmap()(source, prediction)

    assert np.count_nonzero(heatmap.array) == 0


def test_density_to_heatmap_returns_colored_image_for_positive_density() -> None:
    source = ImagePayload(array=np.zeros((12, 16, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32))

    _, heatmap = DensityToHeatmap()(source, prediction)

    assert np.count_nonzero(heatmap.array) > 0


def test_density_to_heatmap_does_not_mutate_prediction() -> None:
    density = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    source = ImagePayload(array=np.zeros((12, 16, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=density.copy())

    DensityToHeatmap()(source, prediction)

    assert np.array_equal(prediction.density_map, density)


def test_density_to_heatmap_accepts_custom_colormap_and_interpolation() -> None:
    source = ImagePayload(array=np.zeros((20, 10, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32))

    _, heatmap = DensityToHeatmap(
        colormap=2,  # cv2.COLORMAP_JET
        interpolation=0,  # cv2.INTER_NEAREST
    )(source, prediction)

    assert heatmap.array.shape == (20, 10, 3)
    assert heatmap.array.dtype == np.uint8


def test_blend_images_preserves_shape_dtype_and_metadata() -> None:
    base = ImagePayload(array=np.zeros((64, 96, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    overlay = ImagePayload(array=np.ones((64, 96, 3), dtype=np.uint8) * 255, color_space="BGR", layout="HWC")

    annotated = BlendImages()(base, overlay)

    assert annotated.array.shape == base.array.shape
    assert annotated.array.dtype == base.array.dtype
    assert annotated.color_space == base.color_space
    assert annotated.layout == base.layout
    assert np.count_nonzero(annotated.array) > 0


def test_blend_images_honors_weights() -> None:
    base = ImagePayload(array=np.zeros((2, 2, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    overlay = ImagePayload(array=np.ones((2, 2, 3), dtype=np.uint8) * 200, color_space="BGR", layout="HWC")

    annotated = BlendImages(base_weight=0.25, overlay_weight=0.75)(base, overlay)

    assert np.all(annotated.array == 150)


def test_blend_images_rejects_non_hwc_layout() -> None:
    base = ImagePayload(array=np.zeros((3, 8, 8), dtype=np.uint8), color_space="BGR", layout="CHW")
    overlay = ImagePayload(array=np.zeros((3, 8, 8), dtype=np.uint8), color_space="BGR", layout="CHW")

    with pytest.raises(ValueError, match="expects HWC images"):
        BlendImages()(base, overlay)


def test_blend_images_rejects_mismatched_shapes() -> None:
    base = ImagePayload(array=np.zeros((8, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    overlay = ImagePayload(array=np.zeros((10, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")

    with pytest.raises(ValueError, match="requires matching shapes"):
        BlendImages()(base, overlay)
