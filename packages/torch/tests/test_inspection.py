from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes.torch import TorchTensorRegistry


def _inspection_tools():
    from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock

    return ImageBlock, TextBlock, PipelineInspector()


def test_pipeline_inspector_formats_torch_tensor_registry_like_tensor_registry():
    ImageBlock, TextBlock, inspector = _inspection_tools()
    registry = TorchTensorRegistry(
        {
            "class_queries_logits": torch.zeros((100, 81), dtype=torch.float32),
            "masks_queries_logits": torch.zeros((100, 96, 96), dtype=torch.float32),
        }
    )

    blocks = inspector._output_to_blocks(registry)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "TorchTensorRegistry"
    assert blocks[0].rows == [
        ("class_queries_logits", "(100, 81)@cpu"),
        ("masks_queries_logits", "(100, 96, 96)@cpu"),
    ]


def test_pipeline_inspector_formats_primitive_tuple_as_single_block():
    ImageBlock, TextBlock, inspector = _inspection_tools()
    blocks = inspector._output_to_blocks((480, 640))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "tuple"
    assert blocks[0].rows == [("", "(480, 640)")]


def test_pipeline_inspector_formats_rgb_ndarray_as_image():
    ImageBlock, TextBlock, inspector = _inspection_tools()
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    blocks = inspector._output_to_blocks(image)

    assert len(blocks) == 2
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ndarray  6×4  RGB"
    assert np.array_equal(blocks[0].array, image)
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].title == "ndarray"
    assert blocks[1].rows == [("shape", "(4, 6, 3)"), ("dtype", "uint8")]


def test_pipeline_inspector_formats_non_image_ndarray_as_text():
    ImageBlock, TextBlock, inspector = _inspection_tools()
    blocks = inspector._output_to_blocks(np.zeros((2, 3), dtype=np.float32))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ndarray"
    assert blocks[0].rows == [("shape", "(2, 3)"), ("dtype", "float32")]


def test_pipeline_inspector_formats_list_of_dicts():
    ImageBlock, TextBlock, inspector = _inspection_tools()
    blocks = inspector._output_to_blocks(
        [
            {"class_id": 1, "score": 0.9},
            {"class_id": 2, "score": 0.8},
        ]
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "list  ×2"
    assert blocks[0].rows == [
        ("[0]", "dict  class_id 1  |  score 0.9"),
        ("[1]", "dict  class_id 2  |  score 0.8"),
    ]
