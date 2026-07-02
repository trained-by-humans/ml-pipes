from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ..region import RegionOpener
from ..tracing import StepSpan, _fmt_batch_size
from .views import (
    ImageBlock,
    OutputBlock,
    OutputFormatter,
    SpanFormatter,
    StepView,
    TextBlock,
    _apply_image_carry,
    _build_span_metadata,
    _make_grid,
)


def _import_inspection_dependencies() -> tuple[object, type, type, type, type, type, type, type, type, type]:
    try:
        cv2 = import_module("cv2")
        from ml_pipes.onnx import RuntimeOutputs
        from ml_pipes.tensor import TensorPayload, TensorRegistry
        from ml_pipes.vision import (
            Detections,
            ImagePayload,
            ResizeTransform,
            Segmentations,
            Tile,
            TileRect,
        )
    except ImportError as exc:  # pragma: no cover - exercised when optional packages are absent
        raise ImportError(
            "ml_pipes.inspection requires the optional inspection extra. "
            "Install it with `pip install ml-pipes[inspection]`."
        ) from exc
    return cv2, Detections, ImagePayload, ResizeTransform, RuntimeOutputs, Segmentations, TensorPayload, TensorRegistry, Tile, TileRect


cv2, Detections, ImagePayload, ResizeTransform, RuntimeOutputs, Segmentations, TensorPayload, TensorRegistry, Tile, TileRect = (
    _import_inspection_dependencies()
)

try:
    from ml_pipes.torch.types import TorchTensorRegistry
except Exception:  # pragma: no cover - ImportError: torch absent; other: torch present but C extension not fully initialised
    TorchTensorRegistry = None


def _fmt_floats(seq: Any, precision: int = 3) -> str:
    try:
        return "(" + ", ".join(f"{x:.{precision}g}" for x in seq) + ")"
    except Exception:
        return str(seq)


def _image_to_rgb(value: ImagePayload) -> np.ndarray:
    arr = value.array
    if value.color_space == "BGR":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def _is_rgb_image_array(value: np.ndarray) -> bool:
    return value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3


def _tensor_item_to_heatmap(arr: np.ndarray, layout: str) -> np.ndarray:
    """Render a single (non-batched) tensor item as a false-colour heatmap."""

    if arr.ndim == 3:
        ch = arr[0] if layout.startswith("C") else arr[:, :, 0]
    else:
        ch = arr
    ch = ch.astype(np.float32)
    mn, mx = ch.min(), ch.max()
    if mx > mn:
        ch = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        ch = np.zeros_like(ch, dtype=np.uint8)
    heat = cv2.applyColorMap(ch, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _format_tile_rect(value: TileRect) -> list[OutputBlock]:
    w = value.x2 - value.x1
    h = value.y2 - value.y1
    return [TextBlock("TileRect", [("origin", f"({value.x1}, {value.y1})"), ("size", f"{w}×{h}")])]


def _format_tiles_with_overlay(
    tiles: list[ImagePayload],
    rects: list[TileRect],
) -> list[OutputBlock]:
    """Tile grid with click-to-toggle coverage map."""

    tint = np.array([0.25, 0.45, 1.0], dtype=np.float32)
    tile_arrays = [_image_to_rgb(tile) for tile in tiles]
    grid = _make_grid(tile_arrays, divider=2)

    h = max(rect.y2 for rect in rects)
    w = max(rect.x2 for rect in rects)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for img, rect in zip(tile_arrays, rects):
        rh, rw = rect.y2 - rect.y1, rect.x2 - rect.x1
        canvas[rect.y1:rect.y2, rect.x1:rect.x2] = cv2.resize(img, (rw, rh))

    coverage = np.zeros((h, w), dtype=np.int32)
    for rect in rects:
        coverage[rect.y1:rect.y2, rect.x1:rect.x2] += 1

    extra = (coverage - 1).clip(0, None).astype(np.float32)
    mx = float(extra.max())
    intensity = (extra / mx if mx > 0 else extra)[:, :, None]
    mult = 1.0 - intensity * (1.0 - tint)
    overlay = (canvas.astype(np.float32) * mult).clip(0, 255).astype(np.uint8)

    n = len(tiles)
    return [
        ImageBlock(
            title=f"ImagePayload  ×{n}  (click to toggle overlap map)",
            array=grid,
            overlay_array=overlay,
        )
    ]


def _region_summary_block(span: StepSpan) -> list[OutputBlock]:
    """Text block summarising a region opener from its child_trace metadata."""

    ct = span.child_trace
    rows: list[tuple[str, str]] = []
    if ct.workers is not None:
        rows.append(("items", _fmt_batch_size(ct.batch_size)))
        rows.append(("concurrency", str(ct.workers)))
        rows.append(("steps", str(len(ct.spans))))
        rows.append(("total", f"{ct.total_duration_s * 1000:.1f} ms"))
    elif ct.batch_size is not None:
        rows.append(("batch size", _fmt_batch_size(ct.batch_size)))
        rows.append(("steps", str(len(ct.spans))))
        rows.append(("total", f"{ct.total_duration_s * 1000:.1f} ms"))
    else:
        rows.append(("steps", str(len(ct.spans))))
        rows.append(("total", f"{ct.total_duration_s * 1000:.1f} ms"))
    return [TextBlock(span.label.split(":", 1)[-1], rows)]


_OUTPUT_FORMATTERS: dict[type, OutputFormatter] | None = None
_SPAN_FORMATTERS: dict[type, SpanFormatter] | None = None


def _register_builtin_formatters() -> tuple[dict[type, OutputFormatter], dict[type, SpanFormatter]]:
    output_formatters: dict[type, OutputFormatter] = {}
    span_formatters: dict[type, SpanFormatter] = {}

    def _format_segmentations(value: Segmentations) -> list[OutputBlock]:
        name = type(value).__name__
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}  mask✓") for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [TextBlock(f"{name} ({n})", rows)]

    def _format_detections(value: Detections) -> list[OutputBlock]:
        name = type(value).__name__
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}") for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [TextBlock(f"{name} ({n})", rows)]

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

    def _format_ndarray(value: np.ndarray) -> list[OutputBlock]:
        if _is_rgb_image_array(value):
            height, width = value.shape[:2]
            return [
                ImageBlock(title=f"ndarray  {width}×{height}  RGB", array=value),
                TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))]),
            ]
        return [TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))])]

    def _format_tensor(value: TensorPayload) -> list[OutputBlock]:
        name = type(value).__name__
        arr = value.array
        if arr.ndim == 4 and value.layout.startswith("N"):
            n = arr.shape[0]
            item_layout = value.layout[1:]
            heatmaps = [_tensor_item_to_heatmap(arr[i], item_layout) for i in range(n)]
            return [ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=_make_grid(heatmaps))]
        return [ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=_tensor_item_to_heatmap(arr, value.layout))]

    def _format_resize_transform(value: ResizeTransform) -> list[OutputBlock]:
        return [TextBlock(type(value).__name__, [
            ("scale", _fmt_floats(value.scale)),
            ("pad", _fmt_floats(value.pad)),
            ("original", str(value.original_shape)),
            ("resized", str(value.resized_shape)),
        ])]

    def _format_runtime_outputs(value: RuntimeOutputs) -> list[OutputBlock]:
        return [TextBlock(type(value).__name__, [(name, str(tensor.array.shape)) for name, tensor in zip(value.names, value.tensors)])]

    def _format_tensor_registry(value: TensorRegistry) -> list[OutputBlock]:
        rows = []
        for name, tensor in value._tensors.items():
            shape = str(tuple(tensor.shape)) if hasattr(tensor, "device") else str(tensor.shape)
            device = getattr(tensor, "device", None)
            rows.append((name, f"{shape}@{device}" if device is not None else shape))
        return [TextBlock(type(value).__name__, rows)]

    def _format_bytes(value: bytes) -> list[OutputBlock]:
        return [TextBlock("bytes", [("size", f"{len(value) / 1024:.1f} KB")])]

    output_formatters[Segmentations] = _format_segmentations
    output_formatters[Detections] = _format_detections
    output_formatters[ImagePayload] = _format_image
    output_formatters[np.ndarray] = _format_ndarray
    output_formatters[TensorPayload] = _format_tensor
    output_formatters[ResizeTransform] = _format_resize_transform
    output_formatters[RuntimeOutputs] = _format_runtime_outputs
    output_formatters[TensorRegistry] = _format_tensor_registry
    if TorchTensorRegistry is not None:
        output_formatters[TorchTensorRegistry] = _format_tensor_registry
    output_formatters[TileRect] = _format_tile_rect
    output_formatters[bytes] = _format_bytes

    def _region_span_formatter(
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        return StepView(span.label, _build_span_metadata(span), _region_summary_block(span)), last_image

    def _tile_span_formatter(
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        value = span.output_value
        raw_blocks = _format_tiles_with_overlay(value[0], value[1]) if value is not None else []
        blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
        return StepView(span.label, span.operator_config, blocks), image_to_carry

    span_formatters[RegionOpener] = _region_span_formatter
    span_formatters[Tile] = _tile_span_formatter
    return output_formatters, span_formatters


def default_output_formatters() -> dict[type, OutputFormatter]:
    global _OUTPUT_FORMATTERS, _SPAN_FORMATTERS

    if _OUTPUT_FORMATTERS is None or _SPAN_FORMATTERS is None:
        _OUTPUT_FORMATTERS, _SPAN_FORMATTERS = _register_builtin_formatters()
    return dict(_OUTPUT_FORMATTERS)


def default_span_formatters() -> dict[type, SpanFormatter]:
    global _SPAN_FORMATTERS

    if _SPAN_FORMATTERS is None:
        default_output_formatters()
    assert _SPAN_FORMATTERS is not None
    return dict(_SPAN_FORMATTERS)
