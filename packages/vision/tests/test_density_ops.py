from __future__ import annotations

import numpy as np

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ClampDensity, DensityToHeatmap, ImagePayload, SumDensity


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


def test_density_to_heatmap_reads_configured_registry_tensor() -> None:
    source = ImagePayload(array=np.zeros((24, 32, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    registry = TensorRegistry({"estimated_density": np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)})

    returned_source, heatmap = DensityToHeatmap(src="estimated_density")(source, registry)

    assert returned_source is source
    assert heatmap.array.shape == source.array.shape
    assert heatmap.color_space == "BGR"
    assert np.any(heatmap.array)
