from __future__ import annotations

import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Store
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload, NMM, Stitch, Tile


def _make_image(width: int, height: int) -> ImagePayload:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return ImagePayload(array=array, color_space="BGR", layout="HWC")


def test_tile_scatter_gather_stitch_pipeline():
    class _PassRegistry:
        def __call__(self, payload: ImagePayload) -> TensorRegistry:
            return TensorRegistry({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            })

    pipeline = Pipeline([
        Tile(slice_wh=(320, 320), overlap_wh=(0, 0)),
        Store("tile_rects", source=1),
        Pick(0),
        Scatter(max_concurrency=2),
        _PassRegistry(),
        Gather(),
        Recall("tile_rects"),
        Stitch("scores", "classes"),
        NMM(),
    ])

    result = pipeline(_make_image(640, 640))

    assert isinstance(result, TensorRegistry)
    assert result["boxes"].shape == (0, 4)
