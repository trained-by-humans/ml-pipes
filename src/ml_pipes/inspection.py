"""Pipeline step-by-step inspection: data preparation and rendering.

PipelineInspector orchestrates the whole process:
  1. Receives an InspectionResult (StepSpan tree recorded by the pipeline).
  2. Walks the spans and converts each one to a StepView using registered
     formatters — this is the intermediate representation.
  3. Passes the StepView tree to a Renderer to produce the final output.

Components
----------
1. StepView / OutputBlock (intermediate representation)
       StepView is the display representation of one pipeline step — a label,
       operator config, and a list of OutputBlocks. An OutputBlock represents
       the step's output value visually: ImageBlock (numpy RGB array),
       TextBlock (key/value rows), and GroupBlock (nested structured output)
       are the built-in kinds, but custom types can be defined and handled by
       a custom Renderer. Renderers consume
       list[StepView] and know nothing about spans or formatters.

2. SpanFormatter
       A span-level override keyed on operator type. When registered for an
       operator, it takes full control of how that operator's StepView is
       built — receives the raw StepSpan and carry-forward last_image, and
       returns a (StepView, last_image) pair.
       Register via: inspector.register_span_formatter(OpType, formatter)

3. OutputFormatter
       A value-level formatter keyed on output type. Receives the output value
       and returns a list of OutputBlocks. More granular than a SpanFormatter
       and more reusable — the same formatter applies to any operator that
       produces that value type, whether it appears directly or inside a list.
       Register via: inspector.register_output_formatter(ValueType, formatter)

4. Renderer
       Anything that implements render(views: list[StepView]) -> Any.
       Pass a renderer to inspector.render(result, renderer) for full control.
       For the built-in renderers there are terminal shorthands on the
       inspector itself: .to_html(), .save_to_html(), .to_plot(),
       .save_to_plot(), .show(), .show_in_browser().
"""

from __future__ import annotations

import base64
import dataclasses
import html as _html
import os
import tempfile
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

_IN_JUPYTER: bool = "get_ipython" in dir(__builtins__) if isinstance(__builtins__, dict) else hasattr(__builtins__, "get_ipython")
_TINT = np.array([0.25, 0.45, 1.0], dtype=np.float32)
_HTML_ORIENTATIONS = ("horizontal", "vertical")

from .inspection_artifacts import InspectionResult, InspectionSerializer
from .tracing import StepSpan, _fmt_batch_size
from .tiling import TileRect
from .types import (
    Detections,
    ImagePayload,
    ResizeTransform,
    RuntimeOutputs,
    Segmentations,
    TensorPayload,
    TensorRegistry,
)

try:
    from .torch.types import TorchTensorRegistry
except Exception:  # pragma: no cover - ImportError: torch absent; other: torch present but C extension not fully initialised
    TorchTensorRegistry = None


# ---------------------------------------------------------------------------
# Public IR — display primitives
# ---------------------------------------------------------------------------

@dataclass
class ImageBlock:
    title: str
    array: np.ndarray      # RGB, uint8, ready for imshow / imencode
    dim: bool = False      # True when carrying forward a previous step's image
    overlay_array: np.ndarray | None = None  # click-to-toggle alternate view


@dataclass
class TextBlock:
    title: str
    rows: list[tuple[str, str]]   # (key, value) pairs, both pre-stringified
    dim: bool = False


@dataclass
class GroupBlock:
    title: str
    children: list["OutputBlock"]
    dim: bool = False


OutputBlock = ImageBlock | TextBlock | GroupBlock


@dataclass
class StepView:
    label: str                          # "3:Resize"
    operator_config: dict[str, Any]
    blocks: list[OutputBlock]
    error: bool = False
    children: list[StepView] = field(default_factory=list)  # non-empty for region operators


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------

class Renderer(Protocol):
    """Anything that can turn a list of StepViews into an output format."""
    def render(self, views: list[StepView]) -> Any: ...


# ---------------------------------------------------------------------------
# Formatter type aliases
# ---------------------------------------------------------------------------

OutputFormatter = Callable[[Any], list[OutputBlock]]
SpanFormatter = Callable[
    [StepSpan, np.ndarray | None],
    tuple[StepView, np.ndarray | None],
]


# ---------------------------------------------------------------------------
# Low-level array helpers (used by formatters)
# ---------------------------------------------------------------------------

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


def _make_grid(images: list[np.ndarray], divider: int = 0) -> np.ndarray:
    """Tile a list of HxWx3 RGB images into a square-ish grid."""
    import math
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = images[0].shape[:2]
    gh = rows * h + divider * (rows - 1)
    gw = cols * w + divider * (cols - 1)
    grid = np.full((gh, gw, 3), 180, dtype=np.uint8)
    for idx, img in enumerate(images):
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        r, c = divmod(idx, cols)
        y = r * (h + divider)
        x = c * (w + divider)
        grid[y:y + h, x:x + w] = img
    return grid


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


# ---------------------------------------------------------------------------
# Output → blocks helpers (stateless, used by PipelineInspector._output_to_blocks)
# ---------------------------------------------------------------------------

def _format_tile_rect(value: TileRect) -> list[OutputBlock]:
    w = value.x2 - value.x1
    h = value.y2 - value.y1
    return [TextBlock("TileRect", [
        ("origin", f"({value.x1}, {value.y1})"),
        ("size",   f"{w}×{h}"),
    ])]


def _format_tiles_with_overlay(
    tiles: list[ImagePayload],
    rects: list[TileRect],
) -> list[OutputBlock]:
    """Tile grid with click-to-toggle coverage map.

    Overlay: blue tint whose intensity multiplies with each additional tile
    covering a pixel (Android overdraw style).
    """
    tile_arrays = [_image_to_rgb(t) for t in tiles]
    grid = _make_grid(tile_arrays, divider=2)

    h = max(r.y2 for r in rects)
    w = max(r.x2 for r in rects)

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
    mult = 1.0 - intensity * (1.0 - _TINT)
    overlay = (canvas.astype(np.float32) * mult).clip(0, 255).astype(np.uint8)

    n = len(tiles)
    return [ImageBlock(
        title=f"ImagePayload  ×{n}  (click to toggle overlap map)",
        array=grid,
        overlay_array=overlay,
    )]


def _block_summary(blocks: list[OutputBlock]) -> str:
    """Collapse a block list to a single short string for list-item rows."""
    parts = []
    for b in blocks:
        if isinstance(b, ImageBlock):
            parts.append(b.title)
        elif isinstance(b, TextBlock):
            summary = b.title
            rows = b.rows[:3] if b.title == "dict" else b.rows[:1]
            if rows:
                row_summaries = [
                    (f"{k} {v}".rstrip() if k else f"{v}")
                    for k, v in rows
                ]
                if summary:
                    summary += "  " + "  |  ".join(row_summaries)
                else:
                    summary = "  |  ".join(row_summaries)
            parts.append(summary)
        else:
            summary = b.title
            child_summaries = [_block_summary([child]) for child in b.children[:2]]
            if child_summaries:
                summary += "  " + "  |  ".join(child_summaries)
            if len(b.children) > 2:
                summary += f"  |  +{len(b.children) - 2} more"
            parts.append(summary)
    return "  |  ".join(parts)


def _is_primitive_tuple(value: tuple[Any, ...]) -> bool:
    primitive_types = (bool, int, float, str, bytes, type(None), np.generic)
    return bool(value) and all(isinstance(item, primitive_types) for item in value)


def _apply_image_carry(
    raw_blocks: list[OutputBlock],
    last_image: np.ndarray | None,
) -> tuple[list[OutputBlock], np.ndarray | None]:
    """Prepend a dimmed carry-forward image when the step has no image output.

    Returns the blocks and the image to carry forward to the next step.
    """
    image_to_carry = _find_image_in_blocks(raw_blocks)
    if image_to_carry is not None:
        return raw_blocks, image_to_carry
    if last_image is not None:
        return [ImageBlock(title="↑ previous", array=last_image, dim=True)] + raw_blocks, last_image
    return raw_blocks, None


def _find_image_in_blocks(blocks: list[OutputBlock]) -> np.ndarray | None:
    for block in blocks:
        if isinstance(block, ImageBlock):
            return block.array
        if isinstance(block, GroupBlock):
            nested = _find_image_in_blocks(block.children)
            if nested is not None:
                return nested
    return None


def _flatten_step_views(views: list[StepView], depth: int = 0) -> list[tuple[StepView, int]]:
    """Pre-order traversal of a StepView tree, each entry paired with its depth."""
    flat = []
    for v in views:
        flat.append((v, depth))
        flat.extend(_flatten_step_views(v.children, depth + 1))
    return flat


def _region_summary_block(span: StepSpan) -> list[OutputBlock]:
    """Text block summarising a region opener from its child_trace metadata."""
    ct = span.child_trace
    rows: list[tuple[str, str]] = []
    if ct.workers is not None:
        rows.append(("items",       _fmt_batch_size(ct.batch_size)))
        rows.append(("concurrency", str(ct.workers)))
        rows.append(("steps",       str(len(ct.spans))))
        rows.append(("total",       f"{ct.total_duration_s * 1000:.1f} ms"))
    elif ct.batch_size is not None:
        rows.append(("batch size", _fmt_batch_size(ct.batch_size)))
        rows.append(("steps",      str(len(ct.spans))))
        rows.append(("total",      f"{ct.total_duration_s * 1000:.1f} ms"))
    else:
        rows.append(("steps", str(len(ct.spans))))
        rows.append(("total", f"{ct.total_duration_s * 1000:.1f} ms"))
    return [TextBlock(span.label.split(":", 1)[-1], rows)]


def _build_span_metadata(span: StepSpan) -> dict[str, Any]:
    metadata = dict(span.operator_config)
    metadata.update({f"attributes.{key}": value for key, value in span.attributes.items()})
    return metadata


def _normalize_html_orientation(orientation: str) -> str:
    normalized = orientation.strip().lower()
    if normalized not in _HTML_ORIENTATIONS:
        raise ValueError(
            f"Invalid HTML orientation: {orientation!r}. "
            f"Expected one of {list(_HTML_ORIENTATIONS)}."
        )
    return normalized


# ---------------------------------------------------------------------------
# Formatter registries and builtin registration
# ---------------------------------------------------------------------------

# Populated once at module load by _register_builtin_formatters(). Never mutated
# after that — PipelineInspector copies these dicts on construction.
_OUTPUT_FORMATTERS: dict[type, OutputFormatter] = {}
_SPAN_FORMATTERS: dict[type, SpanFormatter] = {}


def _register_builtin_formatters() -> None:
    from .tiling import Tile
    from .region import RegionOpener

    # --- Output formatters (subclasses before base classes) ---

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
                TextBlock(
                    "ndarray",
                    [
                        ("shape", str(value.shape)),
                        ("dtype", str(value.dtype)),
                    ],
                ),
            ]
        return [
            TextBlock(
                "ndarray",
                [
                    ("shape", str(value.shape)),
                    ("dtype", str(value.dtype)),
                ],
            )
        ]

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
            ("scale",    _fmt_floats(value.scale)),
            ("pad",      _fmt_floats(value.pad)),
            ("original", str(value.original_shape)),
            ("resized",  str(value.resized_shape)),
        ])]

    def _format_runtime_outputs(value: RuntimeOutputs) -> list[OutputBlock]:
        return [TextBlock(type(value).__name__, [(n, str(t.array.shape)) for n, t in zip(value.names, value.tensors)])]

    def _format_tensor_registry(value: TensorRegistry) -> list[OutputBlock]:
        rows = []
        for name, tensor in value._tensors.items():
            shape = str(tuple(tensor.shape)) if hasattr(tensor, "device") else str(tensor.shape)
            device = getattr(tensor, "device", None)
            rows.append((name, f"{shape}@{device}" if device is not None else shape))
        return [TextBlock(type(value).__name__, rows)]

    def _format_bytes(value: bytes) -> list[OutputBlock]:
        return [TextBlock("bytes", [("size", f"{len(value) / 1024:.1f} KB")])]

    _OUTPUT_FORMATTERS[Segmentations]   = _format_segmentations
    _OUTPUT_FORMATTERS[Detections]      = _format_detections
    _OUTPUT_FORMATTERS[ImagePayload]    = _format_image
    _OUTPUT_FORMATTERS[np.ndarray]      = _format_ndarray
    _OUTPUT_FORMATTERS[TensorPayload]   = _format_tensor
    _OUTPUT_FORMATTERS[ResizeTransform] = _format_resize_transform
    _OUTPUT_FORMATTERS[RuntimeOutputs]  = _format_runtime_outputs
    _OUTPUT_FORMATTERS[TensorRegistry]  = _format_tensor_registry
    if TorchTensorRegistry is not None:
        _OUTPUT_FORMATTERS[TorchTensorRegistry] = _format_tensor_registry
    _OUTPUT_FORMATTERS[TileRect]        = _format_tile_rect
    _OUTPUT_FORMATTERS[bytes]           = _format_bytes

    # --- Span formatters (base classes before specific) ---

    def _region_span_formatter(
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        return StepView(span.label, _build_span_metadata(span), _region_summary_block(span)), last_image

    def _tile_span_formatter(
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        val = span.output_value
        raw_blocks = _format_tiles_with_overlay(val[0], val[1]) if val is not None else []
        blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
        return StepView(span.label, span.operator_config, blocks), image_to_carry

    _SPAN_FORMATTERS[RegionOpener] = _region_span_formatter
    _SPAN_FORMATTERS[Tile]         = _tile_span_formatter


_register_builtin_formatters()


# ---------------------------------------------------------------------------
# PipelineInspector — fluent display object
# ---------------------------------------------------------------------------

class PipelineInspector:
    """Converts an InspectionResult into views and renders them.

    Starts with all built-in formatters pre-registered. Additional formatters
    can be added via the fluent API before calling a terminal method.

    Example::

        result = pipeline.inspect(image)

        # Default display
        PipelineInspector().show(result)

        # Fluent config
        PipelineInspector()
            .register_output_formatter(MyType, my_formatter)
            .register_span_formatter(MyOp, my_span_formatter)
            .save_to_html(result, "report.html")

        # Custom renderer
        PipelineInspector().render(result, MyRenderer())
    """

    def __init__(self) -> None:
        self._output_fmts: dict[type, OutputFormatter] = dict(_OUTPUT_FORMATTERS)
        self._span_fmts: dict[type, SpanFormatter] = dict(_SPAN_FORMATTERS)

    def register_output_formatter(self, type_: type, formatter: OutputFormatter) -> PipelineInspector:
        """Register a formatter for *type_* output values. Returns self for chaining."""
        self._output_fmts[type_] = formatter
        return self

    def register_span_formatter(self, operator_type: type, formatter: SpanFormatter) -> PipelineInspector:
        """Register a span-level formatter for *operator_type*. Returns self for chaining.

        *formatter* signature::

            formatter(span, last_image) -> (StepView, last_image)
        """
        self._span_fmts[operator_type] = formatter
        return self

    def _find_output_formatter(self, value: Any) -> OutputFormatter | None:
        t = type(value)
        return self._output_fmts.get(t) or next(
            (f for rt, f in self._output_fmts.items() if issubclass(t, rt)),
            None,
        )

    def _is_scalar_field_block(self, block: OutputBlock, value: Any) -> bool:
        return (
            isinstance(block, TextBlock)
            and block.title == type(value).__name__
            and len(block.rows) == 1
            and block.rows[0][0] == ""
        )

    def _recursive_reference_block(self, value: Any) -> TextBlock:
        return TextBlock(type(value).__name__, [("", f"<recursive {type(value).__name__}>")])

    def _named_block(
        self,
        name: str,
        value: Any,
        active_ids: set[int],
    ) -> OutputBlock:
        blocks = self._output_to_blocks(value, active_ids)
        if len(blocks) == 1:
            block = blocks[0]
            if isinstance(block, GroupBlock):
                return GroupBlock(
                    title=f"{name}: {block.title}",
                    children=block.children,
                    dim=block.dim,
                )
            if self._is_scalar_field_block(block, value):
                return TextBlock("", [(name, block.rows[0][1])], dim=block.dim)

        return GroupBlock(
            title=f"{name}: {type(value).__name__}",
            children=blocks,
        )

    def _mapping_to_group(
        self,
        title: str,
        value: Mapping[Any, Any],
        active_ids: set[int],
    ) -> GroupBlock:
        items = list(value.items())
        children = [self._named_block(str(key), item, active_ids) for key, item in items[:12]]
        if len(items) > 12:
            children.append(TextBlock("…", [("", f"+{len(items) - 12} more")]))
        return GroupBlock(title=title, children=children)

    def _dataclass_to_group(
        self,
        value: Any,
        title: str | None = None,
        *,
        active_ids: set[int],
    ) -> GroupBlock:
        return GroupBlock(
            title=title or type(value).__name__,
            children=[
                self._named_block(field.name, getattr(value, field.name), active_ids)
                for field in dataclasses.fields(value)
            ],
        )

    def _output_to_blocks(
        self,
        value: Any,
        active_ids: set[int] | None = None,
    ) -> list[OutputBlock]:
        if active_ids is None:
            active_ids = set()

        if isinstance(value, tuple):
            if _is_primitive_tuple(value):
                return [TextBlock(type(value).__name__, [("", str(value))])]
            blocks: list[OutputBlock] = []
            for item in value:
                blocks.extend(self._output_to_blocks(item, active_ids))
            return blocks

        if isinstance(value, list) and value:
            _LIST_MAX = 6
            formatter = self._find_output_formatter(value[0])
            if formatter is not None:
                all_blocks = [formatter(item) for item in value]
                first = all_blocks[0]
                if first and isinstance(first[0], ImageBlock):
                    grid = _make_grid(
                        [b.array for blocks in all_blocks for b in blocks if isinstance(b, ImageBlock)],
                        divider=2,
                    )
                    return [ImageBlock(title=f"{first[0].title.split('  ')[0]}  ×{len(value)}", array=grid)]
                rows = [(f"[{i}]", _block_summary(blocks)) for i, blocks in enumerate(all_blocks[:_LIST_MAX])]
                if len(value) > _LIST_MAX:
                    rows.append(("…", f"+{len(value) - _LIST_MAX} more"))
                return [TextBlock(f"list  ×{len(value)}", rows)]
            if isinstance(value[0], Mapping) or (dataclasses.is_dataclass(value[0]) and not isinstance(value[0], type)):
                rows = [
                    (f"[{i}]", _block_summary(self._output_to_blocks(item, active_ids)))
                    for i, item in enumerate(value[:_LIST_MAX])
                ]
                if len(value) > _LIST_MAX:
                    rows.append(("…", f"+{len(value) - _LIST_MAX} more"))
                return [TextBlock(f"list  ×{len(value)}", rows)]
            item_type = type(value[0]).__name__
            return [TextBlock(f"list[{item_type}]  ×{len(value)}", [("", "…")])]

        formatter = self._find_output_formatter(value)
        if formatter is not None:
            return formatter(value)

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value_id = id(value)
            if value_id in active_ids:
                return [self._recursive_reference_block(value)]
            active_ids.add(value_id)
            try:
                return [self._dataclass_to_group(value, active_ids=active_ids)]
            finally:
                active_ids.remove(value_id)

        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in active_ids:
                return [self._recursive_reference_block(value)]
            active_ids.add(value_id)
            try:
                return [self._mapping_to_group(type(value).__name__, value, active_ids)]
            finally:
                active_ids.remove(value_id)

        name = type(value).__name__
        text = value if isinstance(value, str) else repr(value)
        return [TextBlock(name, [("", text[:120] + ("…" if len(text) > 120 else ""))])]

    def _span_to_view(
        self,
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        if span.error:
            children, _ = self._trace_to_views(span.child_trace, last_image)
            return StepView(span.label, _build_span_metadata(span), [], error=True, children=children), last_image

        op_type = span.operator_type
        formatter = (
            self._span_fmts.get(op_type) or (
                next((f for t, f in self._span_fmts.items() if issubclass(op_type, t)), None)
                if op_type is not None else None
            )
        )
        if formatter is not None:
            view, image_to_carry = formatter(span, last_image)
            children, _ = self._trace_to_views(span.child_trace, image_to_carry)
            return dataclasses.replace(view, children=children), image_to_carry

        raw_blocks = self._output_to_blocks(span.output_value)
        blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
        children, _ = self._trace_to_views(span.child_trace, image_to_carry)
        return StepView(span.label, _build_span_metadata(span), blocks, children=children), image_to_carry

    def _trace_to_views(
        self,
        trace: Any,
        last_image: np.ndarray | None,
    ) -> tuple[list[StepView], np.ndarray | None]:
        if trace is None:
            return [], last_image
        views = []
        for span in trace.spans:
            view, last_image = self._span_to_view(span, last_image)
            views.append(view)
        return views, last_image

    def build_views(self, result: InspectionResult) -> list[StepView]:
        """Convert spans to a display-ready StepView tree."""
        views, _ = self._trace_to_views(result, None)
        return views

    def render(self, result: InspectionResult, renderer: Renderer) -> Any:
        """Pass the view tree to *renderer* and return its output."""
        return renderer.render(self.build_views(result))

    def to_html(self, result: InspectionResult, orientation: str = "horizontal") -> str:
        """Return a self-contained HTML string."""
        return HtmlRenderer(orientation=orientation).render(self.build_views(result))

    def save_to_html(
        self,
        result: InspectionResult,
        path: str | Path,
        orientation: str = "horizontal",
    ) -> Path:
        """Write an HTML report to *path* and return it."""
        return HtmlRenderer(orientation=orientation).save(self.build_views(result), path)

    def to_plot(
        self,
        result: InspectionResult,
        cols: int = 6,
        cell_w: float = 2.6,
        cell_h: float = 3.2,
    ) -> "matplotlib.figure.Figure":
        """Return a matplotlib Figure."""
        return PlotRenderer(cols=cols, cell_w=cell_w, cell_h=cell_h).render(self.build_views(result))

    def save_to_plot(
        self,
        result: InspectionResult,
        path: str | Path,
        cols: int = 6,
        cell_w: float = 2.6,
        cell_h: float = 3.2,
        dpi: int = 150,
    ) -> Path:
        """Save a matplotlib figure to *path* (e.g. "report.png") and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = self.to_plot(result, cols=cols, cell_w=cell_w, cell_h=cell_h)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return out

    def show(
        self,
        result: InspectionResult,
        cols: int = 6,
        orientation: str = "horizontal",
    ) -> None:
        """Display the result.

        In Jupyter: renders the HTML card strip inline.
        In a script/terminal: opens a matplotlib window via plt.show().
        """
        if _IN_JUPYTER:
            from IPython.display import HTML, display
            display(HTML(self.to_html(result, orientation=orientation)))
        else:
            import matplotlib.pyplot as plt
            self.to_plot(result, cols=cols)
            plt.show()

    def show_in_browser(self, result: InspectionResult, orientation: str = "horizontal") -> None:
        """Open the HTML report in the default web browser via a temporary file."""
        fd, tmp = tempfile.mkstemp(suffix=".html", prefix="ml_pipes_inspect_")
        os.close(fd)
        out = Path(tmp)
        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{self.to_html(result, orientation=orientation)}</body></html>",
            encoding="utf-8",
        )
        webbrowser.open(out.as_uri())


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_IMG_STYLE = "max-width:240px;max-height:200px;object-fit:contain;display:block;"
_TBL_STYLE = "font-size:11px;border-collapse:collapse;width:100%;"
_TD_K = "padding:1px 6px 1px 0;color:#555;white-space:nowrap;vertical-align:top;"
_TD_V = "padding:1px 0;word-break:break-all;vertical-align:top;"
_TITLE_STYLE = (
    "font-size:10px;font-weight:600;text-transform:uppercase;"
    "letter-spacing:0.04em;margin-bottom:3px;"
)

_CSS = """
<style>
.insp-container {
  display: flex;
  gap: 10px;
  padding: 10px 4px 14px;
  font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
}
.insp-container--horizontal {
  flex-direction: row;
  overflow-x: auto;
  align-items: flex-start;
}
.insp-container--vertical {
  flex-direction: column;
  overflow-y: auto;
  align-items: stretch;
}
.insp-card {
  min-width: 200px;
  max-width: 260px;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #fff;
  flex-shrink: 0;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.insp-container--vertical .insp-card {
  min-width: 0;
  width: calc(100% - 8px);
  max-width: 1100px;
}
.insp-card-head {
  background: #f0f0f0;
  padding: 5px 8px 4px;
  border-bottom: 1px solid #d8d8d8;
  border-radius: 6px 6px 0 0;
}
.insp-card-name-wrap { display: flex; align-items: center; gap: 5px; }
.insp-card-name {
  font-size: 12px; font-weight: 600; color: #222;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1; min-width: 0;
}
.insp-cfg-icon {
  font-size: 10px; color: #999; cursor: default; line-height: 1;
  border: 1px solid #ccc; border-radius: 3px; padding: 0 3px;
  background: #fff; user-select: none; flex-shrink: 0;
}
.insp-cfg-icon:hover { color: #333; border-color: #888; }
.insp-card-body { padding: 7px 8px; border-radius: 0 0 6px 6px; overflow-x: auto; }
.insp-card-error .insp-card-head { background: #fff0f0; border-color: #f5a0a0; }
.insp-card-error .insp-card-name { color: #c00; }
.insp-container--vertical .insp-card-name { white-space: normal; }
.insp-group {
  margin-bottom: 8px;
}
.insp-group--dim { opacity: 0.6; }
.insp-group-body {
  margin-left: 12px;
}
.insp-group-empty {
  font-size: 11px;
  color: #aaa;
  font-style: italic;
}
.insp-inline-grid {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  column-gap: 6px;
  row-gap: 1px;
  align-items: start;
  font-size: 11px;
  margin-bottom: 4px;
}
.insp-inline-grid--dim { opacity: 0.6; }
.insp-inline-key {
  color: #555;
  white-space: nowrap;
}
.insp-inline-val {
  min-width: 0;
  word-break: break-all;
}
#insp-cfg-popup {
  display: none; position: fixed; z-index: 9999;
  background: #1e1e1e; color: #d4d4d4; border-radius: 5px;
  padding: 8px 10px; font-size: 11px;
  font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
  white-space: nowrap; box-shadow: 0 4px 14px rgba(0,0,0,.4);
  pointer-events: none;
}
#insp-cfg-popup table { border-collapse: collapse; }
#insp-cfg-popup td { padding: 1px 10px 1px 0; vertical-align: top; }
.insp-cfg-key { color: #9cdcfe; }
.insp-cfg-val { color: #ce9178; }

</style>
<div id="insp-cfg-popup"></div>
<script>
(function() {
  var popup = document.getElementById('insp-cfg-popup');
  document.addEventListener('mouseover', function(e) {
    var icon = e.target.closest('.insp-cfg-icon');
    if (!icon) return;
    popup.innerHTML = icon.dataset.cfg;
    popup.style.display = 'block';
    var r = icon.getBoundingClientRect();
    var left = r.left;
    var top = r.bottom + 6;
    var pw = popup.offsetWidth;
    if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
  });
  document.addEventListener('mouseout', function(e) {
    if (e.target.closest('.insp-cfg-icon')) popup.style.display = 'none';
  });
})();
</script>
"""


class HtmlRenderer:
    """Renders a list of StepViews as HTML cards.

    Example::

        views = PipelineInspector().build_views(result)
        html: str = HtmlRenderer().render(views)
        HtmlRenderer().save(views, "report.html")
    """

    def __init__(self, orientation: str = "horizontal") -> None:
        self.orientation = _normalize_html_orientation(orientation)

    def render(self, views: list[StepView]) -> str:
        """Return a self-contained HTML string."""
        cards = [self._render_card(v) for v, _ in _flatten_step_views(views)]
        return (
            f'{_CSS}<div class="insp-container insp-container--{self.orientation}">'
            f'{"".join(cards)}</div>'
        )

    def save(self, views: list[StepView], path: str | Path) -> Path:
        """Write the HTML report to *path* and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{self.render(views)}</body></html>",
            encoding="utf-8",
        )
        return out

    def _render_card(self, view: StepView) -> str:
        error_cls = " insp-card-error" if view.error else ""
        tooltip = self._render_config_tooltip(view.operator_config)
        body = self._render_body(view)
        return (
            f'<div class="insp-card{error_cls}">'
            f'<div class="insp-card-head">'
            f'<div class="insp-card-name-wrap">'
            f'<span class="insp-card-name">{_html.escape(view.label)}</span>'
            f'{tooltip}'
            f'</div></div>'
            f'<div class="insp-card-body">{body}</div>'
            f'</div>'
        )

    def _render_body(self, view: StepView) -> str:
        if view.error:
            return '<div style="font-size:12px;color:#c00;padding:4px 0;">Error during execution</div>'
        if not view.blocks:
            return "<em style='font-size:11px;color:#aaa;'>no value captured</em>"
        return "".join(self._render_block(b) for b in view.blocks)

    @staticmethod
    def _is_inline_row_block(block: OutputBlock) -> bool:
        return isinstance(block, TextBlock) and not block.title

    @staticmethod
    def _leaf_row(block: TextBlock) -> tuple[str, str] | None:
        if len(block.rows) != 1:
            return None
        key, value = block.rows[0]
        if block.title and not key:
            return block.title, value
        if not block.title and key:
            return key, value
        return None

    def _render_inline_rows(self, blocks: list[TextBlock]) -> str:
        if not blocks:
            return ""
        dim = all(block.dim for block in blocks)
        rows = "".join(
            f'<div class="insp-inline-key">{_html.escape(key)}</div>'
            f'<div class="insp-inline-val">{_html.escape(value)}</div>'
            for block in blocks
            for key, value in block.rows
        )
        dim_class = " insp-inline-grid--dim" if dim else ""
        return f'<div class="insp-inline-grid{dim_class}">{rows}</div>'

    def _render_group_children(self, children: list[OutputBlock]) -> str:
        parts: list[str] = []
        inline_rows: list[TextBlock] = []

        def flush_inline_rows() -> None:
            nonlocal inline_rows
            if inline_rows:
                parts.append(self._render_inline_rows(inline_rows))
                inline_rows = []

        for child in children:
            if self._is_inline_row_block(child):
                inline_rows.append(child)
                continue
            flush_inline_rows()
            parts.append(self._render_block(child))

        flush_inline_rows()
        return "".join(parts)

    def _render_block(self, block: OutputBlock) -> str:
        if isinstance(block, ImageBlock):
            dim = block.dim
            title_style = _TITLE_STYLE + ("color:#bbb;" if dim else "color:#555;")
            title_html = f'<div style="{title_style}">{_html.escape(block.title)}</div>'
            _, buf = cv2.imencode(".png", cv2.cvtColor(block.array, cv2.COLOR_RGB2BGR))
            uri = "data:image/png;base64," + base64.b64encode(buf).decode()
            opacity = "0.2" if dim else "1"
            if block.overlay_array is not None:
                _, obuf = cv2.imencode(".png", cv2.cvtColor(block.overlay_array, cv2.COLOR_RGB2BGR))
                overlay_uri = "data:image/png;base64," + base64.b64encode(obuf).decode()
                _TOGGLE_STYLE = (
                    f"{_IMG_STYLE}opacity:{opacity};cursor:pointer;"
                    "border:2px dashed #7aaef5;border-radius:3px;"
                    "transition:border-color .15s;"
                    "box-sizing:border-box;"
                )
                img_html = (
                    f'<img src="{uri}" data-primary="{uri}" data-overlay="{overlay_uri}"'
                    f' style="{_TOGGLE_STYLE}"'
                    f' title="Click to toggle overlap map"'
                    f' onmouseover="this.style.borderColor=\'#3a7de0\';"'
                    f' onmouseout="this.style.borderColor=\'#7aaef5\';"'
                    f' onclick="var s=this.src===this.dataset.primary;'
                    f'this.src=s?this.dataset.overlay:this.dataset.primary;'
                    f'this.style.borderStyle=s?\'solid\':\'dashed\';">'
                )
            else:
                img_html = f'<img src="{uri}" style="{_IMG_STYLE}opacity:{opacity};">'
            return f'<div style="margin-bottom:8px;">{title_html}{img_html}</div>'

        if isinstance(block, GroupBlock):
            dim_class = " insp-group--dim" if block.dim else ""
            dim = block.dim
            title_style = _TITLE_STYLE + ("color:#bbb;" if dim else "color:#555;")
            title_html = f'<div style="{title_style}">{_html.escape(block.title)}</div>'
            children = (
                self._render_group_children(block.children)
                if block.children else
                '<div class="insp-group-empty">empty</div>'
            )
            return (
                f'<div class="insp-group{dim_class}">'
                f'{title_html}'
                f'<div class="insp-group-body">{children}</div>'
                f'</div>'
            )

        dim = block.dim
        leaf_row = self._leaf_row(block)
        if leaf_row is not None:
            return self._render_inline_rows([TextBlock("", [leaf_row], dim=dim)])

        title_style = _TITLE_STYLE + ("color:#bbb;" if dim else "color:#555;")
        title_html = (
            f'<div style="{title_style}">{_html.escape(block.title)}</div>'
            if block.title else ""
        )
        inner = "".join(
            f'<tr><td style="{_TD_K}">{_html.escape(k)}</td>'
            f'<td style="{_TD_V}">{_html.escape(v)}</td></tr>'
            for k, v in block.rows
        )
        tbl = f'<table style="{_TBL_STYLE}">{inner}</table>'
        return f'<div style="margin-bottom:8px;">{title_html}{tbl}</div>'

    def _render_config_tooltip(self, cfg: dict) -> str:
        if not cfg:
            return ""
        rows = "".join(
            f'<tr><td class="insp-cfg-key">{_html.escape(k)}</td>'
            f'<td class="insp-cfg-val">{_html.escape(repr(v))}</td></tr>'
            for k, v in cfg.items()
        )
        table_html = f"<table>{rows}</table>"
        return f'<span class="insp-cfg-icon" data-cfg="{_html.escape(table_html, quote=True)}">⚙</span>'


# ---------------------------------------------------------------------------
# Plot renderer
# ---------------------------------------------------------------------------

class PlotRenderer:
    """Renders a list of StepViews as a matplotlib Figure.

    Example::

        views = PipelineInspector().build_views(result)
        fig = PlotRenderer(cols=4).render(views)
        fig.savefig("steps.png", dpi=150)
    """

    def __init__(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> None:
        self.cols = cols
        self.cell_w = cell_w
        self.cell_h = cell_h

    def _block_to_lines(self, block: OutputBlock, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        if isinstance(block, ImageBlock):
            return [prefix + block.title]

        if isinstance(block, GroupBlock):
            lines = [prefix + block.title]
            if not block.children:
                lines.append(prefix + "  empty")
                return lines
            for child in block.children:
                lines.extend(self._block_to_lines(child, indent + 1))
            return lines

        lines = [prefix + block.title] if block.title else []
        lines.extend(
            prefix + "  " + (f"{key}: {value}" if key else value)
            for key, value in block.rows
        )
        return lines

    def render(self, views: list[StepView]) -> "matplotlib.figure.Figure":
        import matplotlib
        import matplotlib.pyplot as plt

        flat = _flatten_step_views(views)
        n = len(flat)
        rows = max(1, (n + self.cols - 1) // self.cols)
        fig, axes = plt.subplots(rows, self.cols,
                                 figsize=(self.cols * self.cell_w, rows * self.cell_h))
        ax_flat: list[matplotlib.axes.Axes] = np.array(axes).flatten().tolist()

        for i, (view, depth) in enumerate(flat):
            self._render_axes(ax_flat[i], view, depth)
        for ax in ax_flat[n:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.5)
        return fig

    def _render_axes(self, ax: "matplotlib.axes.Axes", view: StepView, depth: int = 0) -> None:
        ax.set_xticks([])
        ax.set_yticks([])

        if depth > 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaa")
                spine.set_linewidth(0.8)
                spine.set_linestyle("dashed")

        if view.error:
            for spine in ax.spines.values():
                spine.set_edgecolor("#c00")
                spine.set_linewidth(2)
                spine.set_linestyle("solid")
            ax.set_title("  " * depth + view.label, fontsize=7.5, fontweight="bold", pad=3, loc="left")
            return

        for block in view.blocks:
            if isinstance(block, ImageBlock):
                ax.imshow(block.array, alpha=0.2 if block.dim else 1.0)
            else:
                text = "\n".join(self._block_to_lines(block))
                ax.text(
                    0.04, 0.97, text,
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=6.5, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
                )

        cfg = view.operator_config
        cfg_short = ""
        if cfg:
            joined = ", ".join(f"{k}={v!r}" for k, v in cfg.items())
            cfg_short = "\n" + (joined if len(joined) <= 30 else joined[:27] + "…")

        ax.set_title(
            "  " * depth + view.label + cfg_short,
            fontsize=7.5, fontweight="bold", pad=3, loc="left",
        )


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------
