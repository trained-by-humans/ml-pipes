from __future__ import annotations

import numpy as np

from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock


def test_pipeline_inspector_formats_primitive_tuple_as_single_block():
    inspector = PipelineInspector()
    blocks = inspector._value_to_blocks((480, 640))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "tuple"
    assert blocks[0].rows == [("", "(480, 640)")]


def test_pipeline_inspector_formats_rgb_ndarray_as_image():
    inspector = PipelineInspector()
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    blocks = inspector._value_to_blocks(image)

    assert len(blocks) == 2
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ndarray  6×4  RGB"
    assert np.array_equal(blocks[0].array, image)
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].title == "ndarray"
    assert blocks[1].rows == [("shape", "(4, 6, 3)"), ("dtype", "uint8")]


def test_pipeline_inspector_formats_non_image_ndarray_as_text():
    inspector = PipelineInspector()
    blocks = inspector._value_to_blocks(np.zeros((2, 3), dtype=np.float32))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ndarray"
    assert blocks[0].rows == [("shape", "(2, 3)"), ("dtype", "float32")]
