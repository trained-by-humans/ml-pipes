from __future__ import annotations

import ml_pipes.onnx as onnx


def test_onnx_component_surface_is_curated() -> None:
    assert onnx.__all__ == [
        "Distribute",
        "Extract",
        "Infer",
        "RuntimeOutputs",
    ]
