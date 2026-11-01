from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml_pipes.inspection import PipelineInspector, TextBlock
from ml_pipes.torch import TensorRegistry


def test_pipeline_inspector_formats_torch_tensor_registry_like_tensor_registry():
    inspector = PipelineInspector()
    registry = TensorRegistry(
        {
            "class_queries_logits": torch.zeros((100, 81), dtype=torch.float32),
            "masks_queries_logits": torch.zeros((100, 96, 96), dtype=torch.float32),
        }
    )

    blocks = inspector._value_to_blocks(registry)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "TensorRegistry"
    assert blocks[0].rows == [
        ("class_queries_logits", "(100, 81)@cpu"),
        ("masks_queries_logits", "(100, 96, 96)@cpu"),
    ]
