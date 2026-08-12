from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock
from ml_pipes.tensor import TensorPayload, TensorRegistry


def test_pipeline_inspector_formats_tensor_registry_from_tensor_package() -> None:
    blocks = PipelineInspector()._output_to_blocks(
        TensorRegistry({"scores": np.zeros((2, 3), dtype=np.float32)})
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "TensorRegistry"
    assert blocks[0].rows[0][0] == "scores"
    assert "(2, 3)" in blocks[0].rows[0][1]


def test_pipeline_inspector_formats_tensor_payload_as_heatmap() -> None:
    pytest.importorskip("cv2")
    payload = TensorPayload(
        array=np.zeros((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    blocks = PipelineInspector()._output_to_blocks(payload)

    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "TensorPayload  (1, 3, 2, 2)  float32"
    assert blocks[0].array.ndim == 3
