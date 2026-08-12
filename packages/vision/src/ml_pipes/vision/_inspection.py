from __future__ import annotations

from typing import Any

import numpy as np

from ml_pipes.inspection._deps import load_cv2
from ml_pipes.inspection._registry import register_step_formatter, register_value_formatter
from ml_pipes.inspection.views import (
    ImageBlock,
    OutputBlock,
    StepView,
    TextBlock,
    _apply_image_carry,
    _make_grid,
)
from ml_pipes.tracing import StepSpan

from .tiling import Tile, TileRect
from .types import Detections, ImagePayload, ResizeTransform, Segmentations


def _fmt_floats(seq: Any, precision: int = 3) -> str:
    try:
        return "(" + ", ".join(f"{x:.{precision}g}" for x in seq) + ")"
    except Exception:
        return str(seq)


def _image_to_rgb(value: ImagePayload) -> np.ndarray:
    if value.color_space != "BGR":
        return value.array
    return load_cv2().cvtColor(value.array, load_cv2().COLOR_BGR2RGB)


def _format_segmentations(value: Segmentations) -> list[OutputBlock]:
    name = type(value).__name__
    count = len(value.boxes)
    rows = [(f"[{index}]", f"cls={value.classes[index]}  score={value.scores[index]:.2f}  mask✓") for index in range(min(count, 6))]
    if count > 6:
        rows.append(("…", f"+{count - 6} more"))
    return [TextBlock(f"{name} ({count})", rows)]


def _format_detections(value: Detections) -> list[OutputBlock]:
    name = type(value).__name__
    count = len(value.boxes)
    rows = [(f"[{index}]", f"cls={value.classes[index]}  score={value.scores[index]:.2f}") for index in range(min(count, 6))]
    if count > 6:
        rows.append(("…", f"+{count - 6} more"))
    return [TextBlock(f"{name} ({count})", rows)]


def _format_image(value: ImagePayload) -> list[OutputBlock]:
    return [
        ImageBlock(
            title=f"{type(value).__name__}  {value.width}×{value.height}  {value.color_space}  {value.layout}",
            array=_image_to_rgb(value),
        ),
        TextBlock(
            type(value).__name__,
            [
                ("shape", str(value.shape)),
                ("spatial_shape", str(value.spatial_shape)),
                ("size", str(value.size)),
                ("dtype", value.dtype),
                ("layout", value.layout),
                ("color_space", value.color_space),
                ("channels", str(value.channels)),
            ],
        ),
    ]


def _format_resize_transform(value: ResizeTransform) -> list[OutputBlock]:
    return [TextBlock(type(value).__name__, [
        ("scale", _fmt_floats(value.scale)),
        ("pad", _fmt_floats(value.pad)),
        ("original", str(value.original_shape)),
        ("resized", str(value.resized_shape)),
    ])]


def _format_tile_rect(value: TileRect) -> list[OutputBlock]:
    width = value.x2 - value.x1
    height = value.y2 - value.y1
    return [TextBlock("TileRect", [("origin", f"({value.x1}, {value.y1})"), ("size", f"{width}×{height}")])]


def _format_tiles_with_overlay(
    tiles: list[ImagePayload],
    rects: list[TileRect],
) -> list[OutputBlock]:
    """Tile grid with click-to-toggle coverage map."""

    cv2 = load_cv2()
    tint = np.array([0.25, 0.45, 1.0], dtype=np.float32)
    tile_arrays = [_image_to_rgb(tile) for tile in tiles]
    grid = _make_grid(tile_arrays, divider=2)

    height = max(rect.y2 for rect in rects)
    width = max(rect.x2 for rect in rects)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for image, rect in zip(tile_arrays, rects):
        rect_height = rect.y2 - rect.y1
        rect_width = rect.x2 - rect.x1
        canvas[rect.y1:rect.y2, rect.x1:rect.x2] = cv2.resize(image, (rect_width, rect_height))

    coverage = np.zeros((height, width), dtype=np.int32)
    for rect in rects:
        coverage[rect.y1:rect.y2, rect.x1:rect.x2] += 1

    extra = (coverage - 1).clip(0, None).astype(np.float32)
    max_extra = float(extra.max())
    intensity = (extra / max_extra if max_extra > 0 else extra)[:, :, None]
    multiplier = 1.0 - intensity * (1.0 - tint)
    overlay = (canvas.astype(np.float32) * multiplier).clip(0, 255).astype(np.uint8)

    return [
        ImageBlock(
            title=f"ImagePayload  ×{len(tiles)}  (click to toggle overlap map)",
            array=grid,
            overlay_array=overlay,
        )
    ]


def _format_tile_step(
    span: StepSpan,
    last_image: np.ndarray | None,
) -> tuple[StepView, np.ndarray | None]:
    value = span.output_value
    raw_blocks = _format_tiles_with_overlay(value[0], value[1]) if value is not None else []
    blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
    return StepView(span.label, span.operator_config, blocks), image_to_carry


def register_inspection_formatters() -> None:
    register_value_formatter(Detections, _format_detections)
    register_value_formatter(ImagePayload, _format_image)
    register_value_formatter(ResizeTransform, _format_resize_transform)
    register_value_formatter(Segmentations, _format_segmentations)
    register_value_formatter(TileRect, _format_tile_rect)
    register_step_formatter(Tile, _format_tile_step)
