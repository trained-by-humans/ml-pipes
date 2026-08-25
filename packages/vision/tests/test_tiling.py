from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload, NMM, Stitch, Tile, TileRect
from ml_pipes.vision.tiling import _compute_tile_rects


def test_tile_rects_exact_fit() -> None:
    rects = _compute_tile_rects(640, 640, (320, 320), (0, 0))
    assert rects == [TileRect(0, 0, 320, 320), TileRect(320, 0, 640, 320), TileRect(0, 320, 320, 640), TileRect(320, 320, 640, 640)]


def test_tile_output_preserves_image_metadata() -> None:
    payload = ImagePayload(array=np.zeros((480, 640, 3), dtype=np.uint8), color_space="RGB", layout="HWC")

    tiles, rects = Tile(slice_wh=(320, 320))(payload)

    assert len(tiles) == len(rects) == 4
    assert all(tile.color_space == "RGB" for tile in tiles)
    assert tiles[-1].array.shape == (160, 320, 3)


def _registry(boxes: list[list[float]], scores: list[float] | None = None, classes: list[int] | None = None) -> TensorRegistry:
    count = len(boxes)
    return TensorRegistry({
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.asarray(scores if scores is not None else [0.9] * count, dtype=np.float32),
        "classes": np.asarray(classes if classes is not None else [0] * count, dtype=np.int32),
    })


def test_stitch_offsets_and_concatenates_tensor_registries() -> None:
    result = Stitch("scores", "classes")(
        [_registry([[10, 10, 50, 50]]), _registry([[5, 5, 30, 30]])],
        [TileRect(0, 0, 320, 320), TileRect(320, 0, 640, 320)],
    )

    assert np.array_equal(result["boxes"], [[10, 10, 50, 50], [325, 5, 350, 30]])
    assert np.allclose(result["scores"], [0.9, 0.9])


def test_stitch_omits_unconfigured_tensors() -> None:
    registry = _registry([[10, 10, 50, 50]])
    registry["masks"] = np.ones((1, 2, 2), dtype=bool)

    result = Stitch("scores", "classes")([registry], [TileRect(0, 0, 320, 320)])

    assert set(result.keys()) == {"boxes", "scores", "classes"}


def test_stitch_concatenates_configured_aligned_tensors() -> None:
    first = _registry([[10, 10, 50, 50]])
    second = _registry([[5, 5, 30, 30]])
    first["embeddings"] = np.array([[1.0, 2.0]], dtype=np.float32)
    second["embeddings"] = np.array([[3.0, 4.0]], dtype=np.float32)

    result = Stitch("scores", "classes", "embeddings")(
        [first, second],
        [TileRect(0, 0, 320, 320), TileRect(320, 0, 640, 320)],
    )

    assert np.allclose(result["embeddings"], [[1.0, 2.0], [3.0, 4.0]])


def test_stitch_rejects_unaligned_configured_tensors() -> None:
    registry = _registry([[10, 10, 50, 50]])
    registry["embeddings"] = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must align with boxes"):
        Stitch("scores", "classes", "embeddings")([registry], [TileRect(0, 0, 320, 320)])


def test_stitch_then_nmm_operates_on_registry_in_place() -> None:
    box = [[10, 10, 100, 100]]
    stitched = Stitch("scores", "classes")(
        [_registry(box, [0.9]), _registry(box, [0.8])],
        [TileRect(0, 0, 640, 640)] * 2,
    )

    result = NMM(iou_threshold=0.5)(stitched)

    assert result is stitched
    assert result["boxes"].shape == (1, 4)
    assert np.allclose(result["scores"], [0.9])


def test_stitch_rejects_mismatched_tile_metadata() -> None:
    with pytest.raises(ValueError, match="one TensorRegistry"):
        Stitch(boxes="boxes")([_registry([])], [])
