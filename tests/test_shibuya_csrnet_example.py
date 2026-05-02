from __future__ import annotations

import numpy as np

from examples.streaming.run_shibuya_csrnet import (
    build_frame_pipeline,
    unwrap_state_dict,
)
from ml_pipes import BlendImages, ClampDensity, DensityPrediction, DensityToHeatmap, ImagePayload, SumDensity, TensorPayload


def test_unwrap_state_dict_strips_module_prefix() -> None:
    checkpoint = {
        "state_dict": {
            "module.frontend.0.weight": 1,
            "module.output_layer.bias": 2,
        }
    }

    assert unwrap_state_dict(checkpoint) == {
        "frontend.0.weight": 1,
        "output_layer.bias": 2,
    }


def test_density_to_heatmap_matches_requested_size() -> None:
    source = ImagePayload(array=np.zeros((24, 32, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    prediction = DensityPrediction(density_map=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32))

    returned_source, heatmap = DensityToHeatmap()(source, prediction)

    assert returned_source is source
    assert heatmap.array.shape == (24, 32, 3)
    assert heatmap.array.dtype == np.uint8


def test_density_prediction_postprocess_clamps_and_sums() -> None:
    prediction = DensityPrediction(
        density_map=np.array([[-1.0, 1.5], [2.0, -0.5]], dtype=np.float32),
    )

    clamped = ClampDensity()(prediction)
    counted = SumDensity()(clamped)

    assert np.array_equal(clamped.density_map, np.array([[0.0, 1.5], [2.0, 0.0]], dtype=np.float32))
    assert counted == 3.5


def test_build_frame_pipeline_returns_source_frame_and_typed_prediction() -> None:
    class StubInfer:
        def __call__(self, tensor: TensorPayload) -> DensityPrediction:
            assert tensor.layout == "NCHW"
            assert tensor.array.shape == (1, 3, 6, 8)
            return DensityPrediction(
                density_map=np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
            )

    pipeline = build_frame_pipeline(StubInfer())
    image = ImagePayload(array=np.zeros((6, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")

    annotated, count = pipeline(image)

    assert isinstance(annotated, ImagePayload)
    assert annotated.array.shape == image.array.shape
    assert count == 8.0


def test_blend_images_preserves_frame_shape() -> None:
    base = ImagePayload(array=np.zeros((64, 96, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    overlay = ImagePayload(array=np.ones((64, 96, 3), dtype=np.uint8) * 255, color_space="BGR", layout="HWC")

    annotated = BlendImages()(base, overlay)

    assert annotated.array.shape == base.array.shape
    assert annotated.array.dtype == base.array.dtype
    assert np.count_nonzero(annotated.array) > 0
