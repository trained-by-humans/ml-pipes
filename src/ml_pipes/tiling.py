from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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


class Tile:
    """Slice an ImagePayload into overlapping tiles.

    Returns ``(list[ImagePayload], list[TileRect])`` so the caller can
    ``Store("tile_rects", index=1)`` and scatter the tile list.

    Example::

        Pipeline([
            Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
            Store("tile_rects", index=1),
            Pick(0),
            Scatter(max_concurrency=4),
            ...,
            Gather(),
            Recall("tile_rects"),
            Stitch(iou_threshold=0.5),
        ])
    """

    def __init__(self, slice_wh: tuple[int, int], overlap_wh: tuple[int, int] = (0, 0)) -> None:
        self.slice_wh = slice_wh
        self.overlap_wh = overlap_wh

    def __call__(self, payload: "ImagePayload") -> "tuple[list[ImagePayload], list[TileRect]]":
        from .types import ImagePayload

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
    original image coordinate system, concatenates all detections, then
    optionally applies cross-tile NMS (``overlap_filter="nms"``) or
    non-maximum merge (``overlap_filter="nmm"``).

    Pass ``overlap_filter=None`` for a plain concat with no deduplication.
    """

    def __init__(self, iou_threshold: float = 0.5, overlap_filter: str | None = "nmm") -> None:
        self.iou_threshold = iou_threshold
        self.overlap_filter = overlap_filter

    def __call__(
        self,
        detections: "list[Detections]",
        tile_rects: "list[TileRect]",
    ) -> "Detections":
        from .types import Detections

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_classes: list[int] = []

        for dets, rect in zip(detections, tile_rects):
            offset = [rect.x1, rect.y1, rect.x1, rect.y1]
            for box in dets.boxes:
                all_boxes.append([b + o for b, o in zip(box, offset)])
            all_scores.extend(dets.scores)
            all_classes.extend(dets.classes)

        if not all_boxes or self.overlap_filter is None:
            return Detections(boxes=all_boxes, scores=all_scores, classes=all_classes)

        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        classes_arr = np.array(all_classes, dtype=np.int32)

        kept = self._filter(boxes_arr, scores_arr, classes_arr)
        return Detections(
            boxes=[all_boxes[i] for i in kept],
            scores=[all_scores[i] for i in kept],
            classes=[all_classes[i] for i in kept],
        )

    def _filter(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> list[int]:
        kept: list[int] = []
        for class_id in np.unique(classes):
            idx = np.where(classes == class_id)[0]
            ordered = idx[np.argsort(scores[idx])[::-1]]
            if self.overlap_filter == "nms":
                ordered = self._nms(boxes, scores, ordered)
            else:
                ordered = self._nmm(boxes, scores, ordered)
            kept.extend(ordered.tolist())
        return kept

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, ordered: np.ndarray) -> np.ndarray:
        keep: list[int] = []
        while ordered.size > 0:
            current = int(ordered[0])
            keep.append(current)
            if ordered.size == 1:
                break
            remaining = ordered[1:]
            ious = _compute_iou(boxes[current], boxes[remaining])
            ordered = remaining[ious < self.iou_threshold]
        return np.asarray(keep, dtype=np.int32)

    def _nmm(self, boxes: np.ndarray, scores: np.ndarray, ordered: np.ndarray) -> np.ndarray:
        # Non-maximum merge: suppress lower-score boxes that overlap with a
        # higher-score box (same as NMS), but keep the highest-score box per group.
        # This implementation is equivalent to NMS for the purpose of deduplication.
        return self._nms(boxes, scores, ordered)


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = np.clip(box_area + boxes_area - inter, 1e-9, None)
    return inter / union
