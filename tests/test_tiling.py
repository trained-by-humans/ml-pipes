"""Tests for TileRect, _compute_tile_rects, Tile, and Stitch."""
from __future__ import annotations

import numpy as np
import pytest

from ml_pipes import (
    Detections,
    Gather,
    ImagePayload,
    NMM,
    Pipeline,
    Pick,
    Recall,
    Scatter,
    Store,
    Stitch,
    Tile,
    TileRect,
)
from ml_pipes.tiling import _compute_tile_rects


# ---------------------------------------------------------------------------
# _compute_tile_rects
# ---------------------------------------------------------------------------

def test_tile_rects_exact_fit():
    # 640×640 image, 320×320 slices, no overlap → 4 tiles exactly
    rects = _compute_tile_rects(640, 640, (320, 320), (0, 0))
    assert len(rects) == 4
    assert TileRect(0, 0, 320, 320) in rects
    assert TileRect(320, 0, 640, 320) in rects
    assert TileRect(0, 320, 320, 640) in rects
    assert TileRect(320, 320, 640, 640) in rects


def test_tile_rects_overlap():
    # 640×320 image, 320×320 slices, 80 overlap on width → 3 tiles wide, 1 tall
    rects = _compute_tile_rects(640, 320, (320, 320), (80, 0))
    # stride_x = 320 - 80 = 240; origins: 0, 240, 480 (last clip to 640)
    assert len(rects) == 3
    assert rects[0] == TileRect(0, 0, 320, 320)
    assert rects[1] == TileRect(240, 0, 560, 320)
    assert rects[2] == TileRect(480, 0, 640, 320)


def test_tile_rects_image_smaller_than_slice():
    rects = _compute_tile_rects(100, 80, (640, 640), (0, 0))
    assert rects == [TileRect(0, 0, 100, 80)]


def test_tile_rects_no_duplicate_boundary():
    # When image is exactly N*slice with no overlap, boundary tile should appear once
    rects = _compute_tile_rects(640, 640, (640, 640), (0, 0))
    assert len(rects) == 1
    assert rects == [TileRect(0, 0, 640, 640)]


# ---------------------------------------------------------------------------
# Tile operator
# ---------------------------------------------------------------------------

def _make_image(w: int, h: int) -> ImagePayload:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    return ImagePayload(array=arr, color_space="BGR", layout="HWC")


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
    assert all(t.color_space == "RGB" for t in tiles)


def test_tile_pixel_content():
    # Paint each quadrant a unique color and verify crops contain the right color.
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :50] = [1, 0, 0]
    arr[:50, 50:] = [0, 2, 0]
    arr[50:, :50] = [0, 0, 3]
    arr[50:, 50:] = [4, 4, 4]
    payload = ImagePayload(array=arr, color_space="BGR", layout="HWC")
    tiles, rects = Tile(slice_wh=(50, 50))(payload)
    assert tiles[0].array[0, 0, 0] == 1   # top-left tile, red channel
    assert tiles[1].array[0, 0, 1] == 2   # top-right tile, green channel


# ---------------------------------------------------------------------------
# Stitch operator
# ---------------------------------------------------------------------------

def _dets(boxes: list[list[float]], scores: list[float] | None = None, classes: list[int] | None = None) -> Detections:
    n = len(boxes)
    return Detections(
        boxes=boxes,
        scores=scores if scores is not None else [0.9] * n,
        classes=classes if classes is not None else [0] * n,
    )


def test_stitch_single_tile_identity():
    # One tile at origin → boxes unchanged
    rect = TileRect(0, 0, 640, 640)
    det = _dets([[10.0, 20.0, 50.0, 60.0]])
    result = Stitch()([det], [rect])
    assert result.boxes == [[10.0, 20.0, 50.0, 60.0]]


def test_stitch_offset_applied():
    rect = TileRect(100, 200, 740, 840)
    det = _dets([[10.0, 20.0, 50.0, 60.0]])
    result = Stitch()([det], [rect])
    assert result.boxes == [[110.0, 220.0, 150.0, 260.0]]


def test_stitch_two_tiles_no_overlap_concat():
    r1 = TileRect(0, 0, 320, 320)
    r2 = TileRect(320, 0, 640, 320)
    d1 = _dets([[10.0, 10.0, 50.0, 50.0]])
    d2 = _dets([[5.0, 5.0, 30.0, 30.0]])
    result = Stitch()([d1, d2], [r1, r2])
    assert len(result.boxes) == 2
    assert [10.0, 10.0, 50.0, 50.0] in result.boxes
    assert [325.0, 5.0, 350.0, 30.0] in result.boxes


def test_stitch_empty_detections():
    r1 = TileRect(0, 0, 640, 640)
    r2 = TileRect(0, 0, 640, 640)
    result = Stitch()([_dets([]), _dets([])], [r1, r2])
    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_nmm_merges_duplicate():
    # Same box detected twice → NMM should merge into one weighted box.
    box = [[10.0, 10.0, 100.0, 100.0]]
    dets = _dets(box + box, scores=[0.9, 0.8])
    result = NMM(iou_threshold=0.5)(dets)
    assert len(result.boxes) == 1
    assert result.scores == [0.9]  # highest score kept


def test_nmm_no_merge_when_no_overlap():
    dets = _dets([[0.0, 0.0, 50.0, 50.0], [200.0, 0.0, 250.0, 50.0]])
    result = NMM(iou_threshold=0.5)(dets)
    assert len(result.boxes) == 2


def test_stitch_then_nmm_removes_duplicate():
    # Same box in two overlapping tiles → Stitch + NMM should keep one.
    r1 = TileRect(0, 0, 640, 640)
    r2 = TileRect(0, 0, 640, 640)
    box = [[10.0, 10.0, 100.0, 100.0]]
    d1 = _dets(box, scores=[0.9])
    d2 = _dets(box, scores=[0.8])
    stitched = Stitch()([d1, d2], [r1, r2])
    result = NMM(iou_threshold=0.5)(stitched)
    assert len(result.boxes) == 1
    assert result.scores == [0.9]


# ---------------------------------------------------------------------------
# Integration: Tile → Scatter → per-tile processing → Gather → Stitch
# ---------------------------------------------------------------------------

def test_tile_scatter_gather_stitch_pipeline():
    class _PassDets:
        """Wrap an ImagePayload as empty Detections (no real model needed)."""
        def __call__(self, payload: ImagePayload) -> Detections:
            return Detections(boxes=[], scores=[], classes=[])

    pipeline = Pipeline([
        Tile(slice_wh=(320, 320), overlap_wh=(0, 0)),
        Store("tile_rects", source=1),
        Pick(0),
        Scatter(max_concurrency=2),
        _PassDets(),
        Gather(),
        Recall("tile_rects"),
        Stitch(),
        NMM(),
    ])

    image = _make_image(640, 640)
    result = pipeline(image)
    assert isinstance(result, Detections)
    assert result.boxes == []
