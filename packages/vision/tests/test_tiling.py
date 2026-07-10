from __future__ import annotations

import numpy as np

from ml_pipes.vision import Detections, ImagePayload, NMM, Stitch, Tile, TileRect
from ml_pipes.vision.tiling import _compute_tile_rects


def test_tile_rects_exact_fit():
    rects = _compute_tile_rects(640, 640, (320, 320), (0, 0))

    assert len(rects) == 4
    assert TileRect(0, 0, 320, 320) in rects
    assert TileRect(320, 0, 640, 320) in rects
    assert TileRect(0, 320, 320, 640) in rects
    assert TileRect(320, 320, 640, 640) in rects


def test_tile_rects_overlap():
    rects = _compute_tile_rects(640, 320, (320, 320), (80, 0))

    assert len(rects) == 3
    assert rects[0] == TileRect(0, 0, 320, 320)
    assert rects[1] == TileRect(240, 0, 560, 320)
    assert rects[2] == TileRect(480, 0, 640, 320)


def test_tile_rects_image_smaller_than_slice():
    rects = _compute_tile_rects(100, 80, (640, 640), (0, 0))

    assert rects == [TileRect(0, 0, 100, 80)]


def test_tile_rects_no_duplicate_boundary():
    rects = _compute_tile_rects(640, 640, (640, 640), (0, 0))

    assert len(rects) == 1
    assert rects == [TileRect(0, 0, 640, 640)]


def _make_image(width: int, height: int) -> ImagePayload:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return ImagePayload(array=array, color_space="BGR", layout="HWC")


def test_tile_output_count():
    payload = _make_image(640, 640)

    tiles, rects = Tile(slice_wh=(320, 320), overlap_wh=(0, 0))(payload)

    assert len(tiles) == 4
    assert len(rects) == 4


def test_tile_output_shapes():
    payload = _make_image(640, 480)

    tiles, rects = Tile(slice_wh=(320, 320), overlap_wh=(0, 0))(payload)

    for tile, rect in zip(tiles, rects):
        expected_h = rect.y2 - rect.y1
        expected_w = rect.x2 - rect.x1
        assert tile.array.shape == (expected_h, expected_w, 3)


def test_tile_preserves_color_space():
    payload = _make_image(100, 100)
    payload = ImagePayload(array=payload.array, color_space="RGB", layout="HWC")

    tiles, _ = Tile(slice_wh=(50, 50))(payload)

    assert all(tile.color_space == "RGB" for tile in tiles)


def test_tile_pixel_content():
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    array[:50, :50] = [1, 0, 0]
    array[:50, 50:] = [0, 2, 0]
    array[50:, :50] = [0, 0, 3]
    array[50:, 50:] = [4, 4, 4]
    payload = ImagePayload(array=array, color_space="BGR", layout="HWC")

    tiles, _ = Tile(slice_wh=(50, 50))(payload)

    assert tiles[0].array[0, 0, 0] == 1
    assert tiles[1].array[0, 0, 1] == 2


def _dets(boxes: list[list[float]], scores: list[float] | None = None, classes: list[int] | None = None) -> Detections:
    count = len(boxes)
    return Detections(
        boxes=boxes,
        scores=scores if scores is not None else [0.9] * count,
        classes=classes if classes is not None else [0] * count,
    )


def test_stitch_single_tile_identity():
    rect = TileRect(0, 0, 640, 640)
    detections = _dets([[10.0, 20.0, 50.0, 60.0]])

    result = Stitch()([detections], [rect])

    assert result.boxes == [[10.0, 20.0, 50.0, 60.0]]


def test_stitch_offset_applied():
    rect = TileRect(100, 200, 740, 840)
    detections = _dets([[10.0, 20.0, 50.0, 60.0]])

    result = Stitch()([detections], [rect])

    assert result.boxes == [[110.0, 220.0, 150.0, 260.0]]


def test_stitch_two_tiles_no_overlap_concat():
    left_rect = TileRect(0, 0, 320, 320)
    right_rect = TileRect(320, 0, 640, 320)
    left_detections = _dets([[10.0, 10.0, 50.0, 50.0]])
    right_detections = _dets([[5.0, 5.0, 30.0, 30.0]])

    result = Stitch()([left_detections, right_detections], [left_rect, right_rect])

    assert len(result.boxes) == 2
    assert [10.0, 10.0, 50.0, 50.0] in result.boxes
    assert [325.0, 5.0, 350.0, 30.0] in result.boxes


def test_stitch_empty_detections():
    first_rect = TileRect(0, 0, 640, 640)
    second_rect = TileRect(0, 0, 640, 640)

    result = Stitch()([_dets([]), _dets([])], [first_rect, second_rect])

    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_stitch_then_nmm_removes_duplicate():
    first_rect = TileRect(0, 0, 640, 640)
    second_rect = TileRect(0, 0, 640, 640)
    box = [[10.0, 10.0, 100.0, 100.0]]
    first_detections = _dets(box, scores=[0.9])
    second_detections = _dets(box, scores=[0.8])

    stitched = Stitch()([first_detections, second_detections], [first_rect, second_rect])
    result = NMM(iou_threshold=0.5)(stitched)

    assert len(result.boxes) == 1
    assert result.scores == [0.9]
