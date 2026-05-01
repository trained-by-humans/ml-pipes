from __future__ import annotations

import numpy as np

from examples.streaming.run_shibuya_csrnet import RollingAverage, density_to_heatmap, render_overlay, unwrap_state_dict


def test_rolling_average_uses_recent_window() -> None:
    avg = RollingAverage(window=3)

    assert avg.update(3) == 3.0
    assert avg.update(6) == 4.5
    assert avg.update(9) == 6.0
    assert avg.update(12) == 9.0


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
    density = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)

    heatmap = density_to_heatmap(density, (24, 32))

    assert heatmap.shape == (24, 32, 3)
    assert heatmap.dtype == np.uint8


def test_render_overlay_preserves_frame_shape() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    density = np.ones((8, 12), dtype=np.float32)

    annotated = render_overlay(frame, density, count=11.0, smoothed=10.5, latency_ms=8.5, fps=12.3)

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert np.count_nonzero(annotated) > 0
