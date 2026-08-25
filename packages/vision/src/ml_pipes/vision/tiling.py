from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.tensor import TensorRegistry
from .types import ImagePayload


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
            Stitch("scores", "classes"),
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


@Operator
class Stitch:
    """Reassemble selected per-tile tensors into one global registry.

    Remaps each tile's box coordinates from tile-local space back to the
    original image coordinate system and concatenates it with the configured
    aligned tensors. Other registry fields are omitted.

    Apply NMS() or NMM() after Stitch to deduplicate cross-tile detections.
    """

    def __init__(self, *srcs: str, boxes: str = "boxes") -> None:
        if boxes in srcs:
            raise ValueError("Stitch boxes must not also be configured as a source tensor")
        if len(set(srcs)) != len(srcs):
            raise ValueError("Stitch source tensor names must be unique")

        self.boxes = boxes
        self.srcs = srcs

    def __call__(
        self,
        registries: list[TensorRegistry],
        tile_rects: list[TileRect],
    ) -> TensorRegistry:
        if len(registries) != len(tile_rects):
            raise ValueError("Stitch requires one TensorRegistry per TileRect")

        if not registries:
            tensors = {self.boxes: np.zeros((0, 4), dtype=np.float32)}
            tensors.update(
                {
                    src: np.zeros((0,), dtype=np.int32 if src == "classes" else np.float32)
                    for src in self.srcs
                }
            )
            return TensorRegistry(tensors)

        box_dtype = registries[0][self.boxes].dtype
        all_tensors = {src: [] for src in self.srcs}
        all_boxes = []
        for registry, rect in zip(registries, tile_rects, strict=True):
            boxes = registry[self.boxes]
            offset = np.asarray([rect.x1, rect.y1, rect.x1, rect.y1], dtype=box_dtype)
            all_boxes.append(boxes + offset)
            for src in self.srcs:
                tensor = registry[src]
                if tensor.shape[0] != boxes.shape[0]:
                    raise ValueError(f"Stitch source {src!r} must align with boxes")
                all_tensors[src].append(tensor)

        tensors = {self.boxes: np.concatenate(all_boxes, axis=0)}
        tensors.update({src: np.concatenate(values, axis=0) for src, values in all_tensors.items()})
        return TensorRegistry(tensors)
