from __future__ import annotations

from dataclasses import dataclass

from .operator import Operator
from .types import Detections, ImagePayload


@dataclass(frozen=True)
class TileRect:
    x1: int
    y1: int
    x2: int
    y2: int


def _compute_tile_rects(
    w: int, h: int, slice_wh: tuple[int, int], overlap_wh: tuple[int, int]
) -> list[TileRect]:
    sw, sh = slice_wh
    ow, oh = overlap_wh
    stride_x = sw - ow
    stride_y = sh - oh
    rects: list[TileRect] = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            x2 = min(x + sw, w)
            y2 = min(y + sh, h)
            rects.append(TileRect(x1=x, y1=y, x2=x2, y2=y2))
            if x2 == w:
                break
            x += stride_x
        if y2 == h:
            break
        y += stride_y
    return rects


@Operator
class Tile:
    """Slice an ImagePayload into overlapping tiles.

    Returns ``(list[ImagePayload], list[TileRect])`` so the caller can
    ``Store("tile_rects", source=1)`` and scatter the tile list.

    Example::

        Pipeline([
            Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
            Store("tile_rects", source=1),
            Pick(0),
            Scatter(max_concurrency=4),
            ...,
            Gather(),
            Recall("tile_rects"),
            Stitch(),
            NMM(),
        ])
    """

    def __init__(self, slice_wh: tuple[int, int], overlap_wh: tuple[int, int] = (0, 0)) -> None:
        self.slice_wh = slice_wh
        self.overlap_wh = overlap_wh

    def __call__(self, payload: "ImagePayload") -> "tuple[list[ImagePayload], list[TileRect]]":
        h, w = payload.array.shape[:2]
        rects = _compute_tile_rects(w, h, self.slice_wh, self.overlap_wh)
        tiles = [
            ImagePayload(
                array=payload.array[r.y1:r.y2, r.x1:r.x2],
                color_space=payload.color_space,
                layout=payload.layout,
            )
            for r in rects
        ]
        return tiles, rects


class Stitch:
    """Reassemble per-tile Detections into a single global Detections.

    Remaps each tile's box coordinates from tile-local space back to the
    original image coordinate system and concatenates all detections.

    Apply NMS() or NMM() after Stitch to deduplicate cross-tile detections.
    """

    def __call__(
        self,
        detections: "list[Detections]",
        tile_rects: "list[TileRect]",
    ) -> "Detections":
        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_classes: list[int] = []

        for dets, rect in zip(detections, tile_rects):
            offset = [rect.x1, rect.y1, rect.x1, rect.y1]
            for box in dets.boxes:
                all_boxes.append([b + o for b, o in zip(box, offset)])
            all_scores.extend(dets.scores)
            all_classes.extend(dets.classes)

        return Detections(boxes=all_boxes, scores=all_scores, classes=all_classes)
