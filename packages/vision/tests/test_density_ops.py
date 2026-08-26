from __future__ import annotations

import numpy as np

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ClampDensity, DensityToHeatmap, ProjectDensity, ResizeTransform, SumDensity


def _registry(density: np.ndarray) -> TensorRegistry:
    return TensorRegistry({"density": density})


def test_density_operators_read_and_write_named_registry_tensors() -> None:
    registry = _registry(np.array([[-1, 2], [3, -4]], dtype=np.int16))

    result = ClampDensity()(registry)

    assert result is registry
    assert np.array_equal(registry["density"], [[0, 2], [3, 0]])
    assert registry["density"].dtype == np.int16
    assert SumDensity()(registry) == 5.0


def test_clamp_density_can_write_to_a_new_name() -> None:
    registry = _registry(np.array([[-1.0, 2.0]], dtype=np.float32))

    ClampDensity(src="density", as_="clamped")(registry)

    assert np.array_equal(registry["density"], [[-1.0, 2.0]])
    assert np.array_equal(registry["clamped"], [[0.0, 2.0]])


def test_density_to_heatmap_applies_turbo_colormap_by_default() -> None:
    registry = TensorRegistry({"estimated_density": np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)})

    heatmap = DensityToHeatmap(src="estimated_density")(registry)

    assert heatmap.array.shape == (2, 2, 3)
    assert heatmap.color_space == "BGR"
    assert heatmap.layout == "HWC"
    assert np.any(heatmap.array)


def test_density_to_heatmap_returns_gray_image_without_a_colormap() -> None:
    registry = TensorRegistry({"estimated_density": np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)})

    heatmap = DensityToHeatmap(src="estimated_density", colormap=None)(registry)

    assert heatmap.array.shape == (2, 2)
    assert heatmap.color_space == "GRAY"
    assert heatmap.layout == "HW"


def test_project_density_removes_letterbox_padding_and_preserves_sum() -> None:
    registry = _registry(np.array([[0.0, 0.0], [1.0, 3.0]], dtype=np.float32))
    transform = ResizeTransform(
        scale=(1.0, 1.0),
        pad=(0.0, 1.0),
        original_shape=(1, 2),
        resized_shape=(2, 2),
    )

    ProjectDensity()(registry, transform)

    assert registry["density"].shape == (1, 2)
    assert np.isclose(registry["density"].sum(), 4.0)
