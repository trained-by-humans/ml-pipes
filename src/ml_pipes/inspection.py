"""Pipeline step-by-step inspection: data preparation and rendering.

Architecture
------------
1. _to_step_view(span, last_image)
       Converts a StepSpan into a _StepView — display-ready primitives:
       a list of _Block objects, where each block is either an image block
       (numpy RGB array) or a text block (title + key/value rows).
       All type-specific logic lives here. Renderers are dumb.

2. HtmlRenderer / PlotRenderer
       Consume _StepView lists and produce HTML or a matplotlib Figure.
       Both classes accept layout/style options; adding a new renderer
       requires no changes to _to_step_view.
"""

from __future__ import annotations

import base64
import dataclasses
import html as _html
import io
import os
import pickle
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .tracing import InvocationTrace, StepSpan, TraceCollector, _fmt_batch_size
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


# ---------------------------------------------------------------------------
# Capture collector (used by Pipeline.inspect)
# ---------------------------------------------------------------------------

class _CaptureCollector(TraceCollector):
    def __init__(self) -> None:
        self.trace: InvocationTrace | None = None

    def on_trace(self, trace: InvocationTrace) -> None:
        self.trace = trace


# ---------------------------------------------------------------------------
# Display primitives
# ---------------------------------------------------------------------------

@dataclass
class _ImageBlock:
    title: str
    array: np.ndarray      # RGB, uint8, ready for imshow / imencode
    dim: bool = False      # True when carrying forward a previous step's image
    overlay_array: np.ndarray | None = None  # click-to-toggle alternate view


@dataclass
class _TextBlock:
    title: str
    rows: list[tuple[str, str]]   # (key, value) pairs, both pre-stringified
    dim: bool = False


_Block = _ImageBlock | _TextBlock


@dataclass
class _StepView:
    label: str                          # "3:Resize"
    operator_config: dict[str, Any]
    blocks: list[_Block]
    error: bool = False
    children: list[_StepView] = field(default_factory=list)  # non-empty for Batch/Scatter regions


# ---------------------------------------------------------------------------
# Value → display primitives
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


def _make_grid(images: list[np.ndarray], divider: int = 0) -> np.ndarray:
    """Tile a list of HxWx3 RGB images into a square-ish grid."""
    import math
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = images[0].shape[:2]
    gh = rows * h + divider * (rows - 1)
    gw = cols * w + divider * (cols - 1)
    grid = np.full((gh, gw, 3), 180, dtype=np.uint8)  # 180 = light grey divider colour
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


def _render_tile_rect(value: TileRect) -> list[_Block]:
    w = value.x2 - value.x1
    h = value.y2 - value.y1
    return [_TextBlock("TileRect", [
        ("origin", f"({value.x1}, {value.y1})"),
        ("size",   f"{w}×{h}"),
    ])]


def _render_tiles_with_overlay(
    tiles: list[ImagePayload],
    rects: list[TileRect],
) -> list[_Block]:
    """Tile grid with click-to-toggle coverage map.

    Overlay: white background, single blue tint whose intensity multiplies
    with each additional tile covering a pixel (Android overdraw style).
    """
    tile_arrays = [_image_to_rgb(t) for t in tiles]
    grid = _make_grid(tile_arrays, divider=2)

    h = max(r.y2 for r in rects)
    w = max(r.x2 for r in rects)

    # Reconstruct source by painting each tile back at its original position.
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for img, rect in zip(tile_arrays, rects):
        rh, rw = rect.y2 - rect.y1, rect.x2 - rect.x1
        canvas[rect.y1:rect.y2, rect.x1:rect.x2] = cv2.resize(img, (rw, rh))

    coverage = np.zeros((h, w), dtype=np.int32)
    for rect in rects:
        coverage[rect.y1:rect.y2, rect.x1:rect.x2] += 1

    # Multiply blend: pixels covered once are untouched; each additional layer
    # multiplies in the tint color, growing progressively more saturated.
    extra = (coverage - 1).clip(0, None).astype(np.float32)
    mx = float(extra.max())
    intensity = (extra / mx if mx > 0 else extra)[:, :, None]
    _TINT = np.array([0.25, 0.45, 1.0], dtype=np.float32)   # blue channel dominates
    mult = 1.0 - intensity * (1.0 - _TINT)
    overlay = (canvas.astype(np.float32) * mult).clip(0, 255).astype(np.uint8)

    n = len(tiles)
    return [_ImageBlock(
        title=f"ImagePayload  ×{n}  (click to toggle overlap map)",
        array=grid,
        overlay_array=overlay,
    )]


# ---------------------------------------------------------------------------
# Block renderer registry
# ---------------------------------------------------------------------------

# Maps a type to a callable (value) -> list[_Block].
# Checked in registration order; first match wins (MRO not respected —
# register subclasses before base classes if both need distinct handlers).
# External code can extend this via register_block_renderer().
_BLOCK_RENDERERS: dict[type, Any] = {}


def _block_summary(blocks: list[_Block]) -> str:
    """Collapse a block list to a single short string for use as a list-item row value."""
    parts = []
    for b in blocks:
        if isinstance(b, _ImageBlock):
            parts.append(b.title)
        else:
            # title + first row value if present
            summary = b.title
            if b.rows:
                k, v = b.rows[0]
                summary += f"  {k} {v}".rstrip() if k else f"  {v}"
            parts.append(summary)
    return "  |  ".join(parts)


def register_block_renderer(type_: type, renderer: Any) -> None:
    """Register a renderer for *type_* in the inspection block registry.

    *renderer* must be callable as ``renderer(value) -> list[_Block]``.
    The new entry takes priority over existing entries for the same type.
    """
    _BLOCK_RENDERERS[type_] = renderer


def _is_tile_output(value: Any) -> bool:
    return (
        isinstance(value, tuple) and len(value) == 2
        and isinstance(value[0], list) and bool(value[0]) and isinstance(value[0][0], ImagePayload)
        and isinstance(value[1], list) and bool(value[1]) and isinstance(value[1][0], TileRect)
    )


def _value_to_blocks(value: Any) -> list[_Block]:
    """Convert a pipeline output value to a list of display blocks."""
    if isinstance(value, tuple):
        blocks: list[_Block] = []
        for item in value:
            blocks.extend(_value_to_blocks(item))
        return blocks

    if isinstance(value, list) and value:
        _LIST_MAX = 6
        for type_, renderer in _BLOCK_RENDERERS.items():
            if isinstance(value[0], type_):
                all_blocks = [renderer(item) for item in value]
                first = all_blocks[0]
                if first and isinstance(first[0], _ImageBlock):
                    grid = _make_grid([b.array for blocks in all_blocks for b in blocks if isinstance(b, _ImageBlock)], divider=2)
                    return [_ImageBlock(title=f"{first[0].title.split('  ')[0]}  ×{len(value)}", array=grid)]
                rows = [(f"[{i}]", _block_summary(blocks)) for i, blocks in enumerate(all_blocks[:_LIST_MAX])]
                if len(value) > _LIST_MAX:
                    rows.append(("…", f"+{len(value) - _LIST_MAX} more"))
                return [_TextBlock(f"list  ×{len(value)}", rows)]
        item_type = type(value[0]).__name__
        return [_TextBlock(f"list[{item_type}]  ×{len(value)}", [("", "…")])]

    for type_, renderer in _BLOCK_RENDERERS.items():
        if isinstance(value, type_):
            return renderer(value)

    name = type(value).__name__
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [_TextBlock(name, [(f.name, str(getattr(value, f.name))) for f in dataclasses.fields(value)])]

    text = repr(value)
    return [_TextBlock(name, [("", text[:120] + ("…" if len(text) > 120 else ""))])]


def _register_builtin_renderers() -> None:
    def _render_image(value: ImagePayload) -> list[_Block]:
        h, w = value.array.shape[:2]
        name = type(value).__name__
        return [_ImageBlock(title=f"{name}  {w}×{h}  {value.color_space}", array=_image_to_rgb(value))]

    def _render_tensor(value: TensorPayload) -> list[_Block]:
        name = type(value).__name__
        arr = value.array
        is_batched = arr.ndim == 4 and value.layout.startswith("N")
        if is_batched:
            n = arr.shape[0]
            item_layout = value.layout[1:]
            heatmaps = [_tensor_item_to_heatmap(arr[i], item_layout) for i in range(n)]
            grid = _make_grid(heatmaps)
            return [_ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=grid)]
        return [_ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=_tensor_item_to_heatmap(arr, value.layout))]

    def _render_resize_transform(value: ResizeTransform) -> list[_Block]:
        name = type(value).__name__
        return [_TextBlock(name, [
            ("scale",    _fmt_floats(value.scale)),
            ("pad",      _fmt_floats(value.pad)),
            ("original", str(value.original_shape)),
            ("resized",  str(value.resized_shape)),
        ])]

    def _render_runtime_outputs(value: RuntimeOutputs) -> list[_Block]:
        name = type(value).__name__
        return [_TextBlock(name, [(n, str(t.array.shape)) for n, t in zip(value.names, value.tensors)])]

    def _render_tensor_registry(value: TensorRegistry) -> list[_Block]:
        name = type(value).__name__
        return [_TextBlock(name, [(k, str(v.shape)) for k, v in value._tensors.items()])]

    def _render_segmentations(value: Segmentations) -> list[_Block]:
        name = type(value).__name__
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}  mask✓") for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [_TextBlock(f"{name} ({n})", rows)]

    def _render_detections(value: Detections) -> list[_Block]:
        name = type(value).__name__
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}") for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [_TextBlock(f"{name} ({n})", rows)]

    def _render_bytes(value: bytes) -> list[_Block]:
        return [_TextBlock("bytes", [("size", f"{len(value) / 1024:.1f} KB")])]

    # Subclasses before base classes so isinstance matching is correct.
    register_block_renderer(Segmentations, _render_segmentations)
    register_block_renderer(Detections,    _render_detections)
    register_block_renderer(ImagePayload,  _render_image)
    register_block_renderer(TensorPayload, _render_tensor)
    register_block_renderer(ResizeTransform,   _render_resize_transform)
    register_block_renderer(RuntimeOutputs,    _render_runtime_outputs)
    register_block_renderer(TensorRegistry,    _render_tensor_registry)
    register_block_renderer(TileRect,          _render_tile_rect)
    register_block_renderer(bytes,             _render_bytes)


_register_builtin_renderers()


def _region_summary_block(span: StepSpan) -> list[_Block]:
    """Text block summarising a region opener (Scatter/Batch) from its child_trace metadata."""
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
    return [_TextBlock(span.label.split(":", 1)[-1], rows)]


def _to_step_view(span: StepSpan, last_image: np.ndarray | None) -> tuple[_StepView, np.ndarray | None]:
    """Convert a StepSpan to a _StepView and return the updated last_image."""
    if span.error:
        children, _ = _build_views_from_trace(span.child_trace, last_image)
        return _StepView(span.label, span.operator_config, [], error=True, children=children), last_image

    val = span.output_value
    if val is None:
        children, _ = _build_views_from_trace(span.child_trace, last_image)
        blocks = _region_summary_block(span) if span.child_trace is not None else []
        return _StepView(span.label, span.operator_config, blocks, children=children), last_image

    is_tile_op = span.label.split(":", 1)[-1] == "Tile"
    if is_tile_op and _is_tile_output(val):
        raw_blocks = _render_tiles_with_overlay(val[0], val[1])
    else:
        raw_blocks = _value_to_blocks(val)

    new_image: np.ndarray | None = None
    for b in raw_blocks:
        if isinstance(b, _ImageBlock):
            new_image = b.array
            break

    blocks: list[_Block]
    if new_image is not None:
        last_image = new_image
        blocks = raw_blocks
    elif last_image is not None:
        carry = _ImageBlock(title="↑ previous", array=last_image, dim=True)
        blocks = [carry] + raw_blocks
    else:
        blocks = raw_blocks

    children, _ = _build_views_from_trace(span.child_trace, last_image)
    return _StepView(span.label, span.operator_config, blocks, children=children), last_image


def _build_views_from_trace(trace: Any, last_image: np.ndarray | None) -> tuple[list[_StepView], np.ndarray | None]:
    if trace is None:
        return [], last_image
    views = []
    for span in trace.spans:
        view, last_image = _to_step_view(span, last_image)
        views.append(view)
    return views, last_image


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
  flex-direction: row;
  overflow-x: auto;
  gap: 10px;
  padding: 10px 4px 14px;
  font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
  align-items: flex-start;
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
.insp-card-body { padding: 7px 8px; border-radius: 0 0 6px 6px; }
.insp-card-error .insp-card-head { background: #fff0f0; border-color: #f5a0a0; }
.insp-card-error .insp-card-name { color: #c00; }
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
    """Renders an InspectionResult as an HTML card strip.

    Example::

        result = pipeline.inspect(image_path)
        renderer = HtmlRenderer()
        html: str = renderer.render(result)          # HTML string — embed anywhere
        renderer.save(result, "report.html")         # write to file
    """

    def render(self, result: "InspectionResult") -> str:
        """Return a self-contained HTML string."""
        cards = [self._card(v) for v in self._flatten(result.build_views())]
        return f'{_CSS}<div class="insp-container">{"".join(cards)}</div>'

    @staticmethod
    def _flatten(views: list[_StepView]) -> list[_StepView]:
        out = []
        for v in views:
            out.append(v)
            out.extend(HtmlRenderer._flatten(v.children))
        return out

    def save(self, result: "InspectionResult", path: str | Path) -> Path:
        """Write the HTML report to *path* and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{self.render(result)}</body></html>",
            encoding="utf-8",
        )
        return out

    def _card(self, view: _StepView) -> str:
        error_cls = " insp-card-error" if view.error else ""
        tooltip = self._config_tooltip(view.operator_config)
        body = self._body(view)
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

    def _body(self, view: _StepView) -> str:
        if view.error:
            return '<div style="font-size:12px;color:#c00;padding:4px 0;">Error during execution</div>'
        if not view.blocks:
            return "<em style='font-size:11px;color:#aaa;'>no value captured</em>"
        return "".join(self._block(b) for b in view.blocks)

    def _block(self, block: _Block) -> str:
        dim = block.dim
        title_style = _TITLE_STYLE + ("color:#bbb;" if dim else "color:#555;")
        title_html = f'<div style="{title_style}">{_html.escape(block.title)}</div>'

        if isinstance(block, _ImageBlock):
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

        # _TextBlock
        inner = "".join(
            f'<tr><td style="{_TD_K}">{_html.escape(k)}</td>'
            f'<td style="{_TD_V}">{_html.escape(v)}</td></tr>'
            for k, v in block.rows
        )
        tbl = f'<table style="{_TBL_STYLE}">{inner}</table>'
        return f'<div style="margin-bottom:8px;">{title_html}{tbl}</div>'

    def _config_tooltip(self, cfg: dict) -> str:
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
    """Renders an InspectionResult as a matplotlib Figure.

    Example::

        result = pipeline.inspect(image_path)
        fig = PlotRenderer(cols=4).render(result)
        fig.savefig("steps.png", dpi=150)
        fig.show()
    """

    def __init__(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> None:
        self.cols = cols
        self.cell_w = cell_w
        self.cell_h = cell_h

    @staticmethod
    def _flatten_views(views: list[_StepView], depth: int = 0) -> list[tuple[_StepView, int]]:
        flat = []
        for v in views:
            flat.append((v, depth))
            flat.extend(PlotRenderer._flatten_views(v.children, depth + 1))
        return flat

    def render(self, result: "InspectionResult") -> "matplotlib.figure.Figure":
        import matplotlib
        import matplotlib.pyplot as plt

        views = self._flatten_views(result.build_views())
        n = len(views)
        rows = max(1, (n + self.cols - 1) // self.cols)
        fig, axes = plt.subplots(rows, self.cols,
                                 figsize=(self.cols * self.cell_w, rows * self.cell_h))
        ax_flat: list[matplotlib.axes.Axes] = np.array(axes).flatten().tolist()

        for i, (view, depth) in enumerate(views):
            self._axes(ax_flat[i], view, depth)
        for ax in ax_flat[n:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.5)
        return fig

    def _axes(self, ax: "matplotlib.axes.Axes", view: _StepView, depth: int = 0) -> None:
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
            if isinstance(block, _ImageBlock):
                ax.imshow(block.array, alpha=0.2 if block.dim else 1.0)
            else:
                text = block.title + "\n" + "\n".join(
                    f"  {k}  {v}" if k else f"  {v}" for k, v in block.rows
                )
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

class InspectionSerializer:
    """Serializes / deserializes an InspectionResult to bytes via pickle.

    Use this when you want to produce bytes in memory and decide yourself
    where they go — S3, a database, a socket, a test fixture, etc.

    Example::

        # On the inference machine:
        result = pipeline.inspect(image_path)
        data: bytes = InspectionSerializer().dumps(result)
        upload_to_s3(data, key="run42/inspection.pkl")

        # On a dev laptop:
        data = download_from_s3(key="run42/inspection.pkl")
        result = InspectionSerializer().loads(data)
        result.show()
    """

    def dumps(self, result: "InspectionResult") -> bytes:
        """Serialize *result* to bytes."""
        return pickle.dumps(result)

    def loads(self, data: bytes) -> "InspectionResult":
        """Deserialize bytes produced by :meth:`dumps` (in-memory counterpart to :meth:`load`)."""
        obj = pickle.loads(data)
        if not isinstance(obj, InspectionResult):
            raise TypeError(f"Expected InspectionResult, got {type(obj).__name__}")
        return obj

    def dump(self, result: "InspectionResult", path: str | Path) -> Path:
        """Serialize *result* to a file. Returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.dumps(result))
        return out

    def load(self, path: str | Path) -> "InspectionResult":
        """Deserialize from a file produced by :meth:`dump`."""
        return self.loads(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------

class InspectionResult:
    """The result of Pipeline.inspect(): one entry per executed step."""

    def __init__(self, spans: list[StepSpan]) -> None:
        self.spans = spans

    def build_views(self) -> list[_StepView]:
        """Convert spans to display-ready _StepView objects.

        Called by HtmlRenderer and PlotRenderer; also available for custom renderers.
        Region spans (Batch, Scatter) produce a _StepView with .children populated.
        """
        views: list[_StepView] = []
        last_image: np.ndarray | None = None
        for span in self.spans:
            view, last_image = _to_step_view(span, last_image)
            views.append(view)
        return views

    def __repr__(self) -> str:
        lines = ["InspectionResult:"]
        self._repr_spans(self.spans, lines, indent=2)
        return "\n".join(lines)

    @staticmethod
    def _repr_spans(spans: list[StepSpan], lines: list[str], indent: int) -> None:
        prefix = " " * indent
        for span in spans:
            shape = span.output_shape or ""
            err = " [ERROR]" if span.error else ""
            lines.append(f"{prefix}{span.label:35s}  {str(shape):20s}{err}")
            if span.child_trace is not None:
                InspectionResult._repr_spans(span.child_trace.spans, lines, indent + 2)

    def _repr_html_(self) -> str:
        return HtmlRenderer().render(self)

    def plot(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> "matplotlib.figure.Figure":
        """Convenience: render with PlotRenderer. Returns the Figure."""
        return PlotRenderer(cols=cols, cell_w=cell_w, cell_h=cell_h).render(self)

    def show(self, cols: int = 6) -> None:
        """Display the result immediately.

        In Jupyter: renders the HTML card strip inline (no browser needed).
        In a script/terminal: opens a matplotlib window via plt.show().
        """
        try:
            get_ipython  # type: ignore[name-defined]  # noqa: F821
            in_jupyter = True
        except NameError:
            in_jupyter = False

        if in_jupyter:
            from IPython.display import HTML, display
            display(HTML(self._repr_html_()))
        else:
            import matplotlib.pyplot as plt
            self.plot(cols=cols)
            plt.show()

    def show_in_browser(self) -> None:
        """Open the HTML report in the default web browser via a temporary file."""
        fd, tmp = tempfile.mkstemp(suffix=".html", prefix="ml_pipes_inspect_")
        os.close(fd)
        out = Path(tmp)
        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{HtmlRenderer().render(self)}</body></html>",
            encoding="utf-8",
        )
        webbrowser.open(out.as_uri())

    def dump(self, path: str | Path) -> Path:
        """Serialize this result to a file. Convenience for InspectionSerializer().dump()."""
        return InspectionSerializer().dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "InspectionResult":
        """Load a serialized result from a file. Convenience for InspectionSerializer().load()."""
        return InspectionSerializer().load(path)
