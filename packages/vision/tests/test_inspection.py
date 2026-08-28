from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from ml_pipes.core import Pipeline
from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock
from ml_pipes.vision import ImagePayload, Tile


def test_pipeline_inspector_formats_image_payload_from_vision_package() -> None:
    payload = ImagePayload(
        array=np.zeros((4, 6, 3), dtype=np.uint8),
        color_space="BGR",
        layout="HWC",
    )

    blocks = PipelineInspector()._value_to_blocks(payload)

    assert len(blocks) == 2
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ImagePayload  6×4  BGR  HWC"
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].title == "ImagePayload"
    assert blocks[1].rows == [
        ("shape", "(4, 6, 3)"),
        ("spatial_shape", "(4, 6)"),
        ("size", "(6, 4)"),
        ("dtype", "uint8"),
        ("layout", "HWC"),
        ("color_space", "BGR"),
        ("channels", "3"),
    ]


def test_pipeline_inspector_formats_tile_span_with_overlay() -> None:
    pipeline = Pipeline([Tile(slice_wh=(2, 2), overlap_wh=(1, 1))])
    payload = ImagePayload(
        array=np.zeros((3, 3, 3), dtype=np.uint8),
        color_space="RGB",
        layout="HWC",
    )

    result = pipeline.inspect(payload)
    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert len(views[0].blocks) == 1
    assert isinstance(views[0].blocks[0], ImageBlock)
    assert views[0].blocks[0].title.startswith("ImagePayload  ×")
    assert views[0].blocks[0].overlay_array is not None


def test_tile_inspection_grid_follows_tile_rect_geometry() -> None:
    pipeline = Pipeline([Tile(slice_wh=(4, 4), overlap_wh=(2, 2))])
    payload = ImagePayload(
        array=np.zeros((12, 18, 3), dtype=np.uint8),
        color_space="RGB",
        layout="HWC",
    )

    views = PipelineInspector().build_views(pipeline.inspect(payload))

    block = views[0].blocks[0]
    assert isinstance(block, ImageBlock)
    # Eight source columns by five rows, with two-pixel dividers.
    assert block.array.shape == (28, 46, 3)
    assert block.overlay_array is not None
    assert block.overlay_array.shape == (12, 18, 3)
