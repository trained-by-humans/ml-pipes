from __future__ import annotations

import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Store
from ml_pipes.vision import Detections, ImagePayload, NMM, Stitch, Tile


def _make_image(width: int, height: int) -> ImagePayload:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return ImagePayload(array=array, color_space="BGR", layout="HWC")


def test_tile_scatter_gather_stitch_pipeline():
    class _PassDetections:
        def __call__(self, payload: ImagePayload) -> Detections:
            return Detections(boxes=[], scores=[], classes=[])

    pipeline = Pipeline([
        Tile(slice_wh=(320, 320), overlap_wh=(0, 0)),
        Store("tile_rects", source=1),
        Pick(0),
        Scatter(max_concurrency=2),
        _PassDetections(),
        Gather(),
        Recall("tile_rects"),
        Stitch(),
        NMM(),
    ])

    result = pipeline(_make_image(640, 640))

    assert isinstance(result, Detections)
    assert result.boxes == []
