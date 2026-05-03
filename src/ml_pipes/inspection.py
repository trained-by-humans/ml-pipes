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
import os
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .tracing import InvocationTrace, StepSpan, TraceCollector
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


def _tensor_to_heatmap(value: TensorPayload) -> np.ndarray:
    arr = value.array
    if arr.ndim == 4:
        ch = arr[0, 0] if value.layout.startswith("N") else arr[0, :, :, 0]
    elif arr.ndim == 3:
        ch = arr[0] if value.layout.startswith("C") else arr[:, :, 0]
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


def _value_to_blocks(value: Any) -> list[_Block]:
    """Convert a pipeline output value to a list of display blocks."""
    if isinstance(value, tuple):
        blocks: list[_Block] = []
        for item in value:
            blocks.extend(_value_to_blocks(item))
        return blocks

    name = type(value).__name__

    if isinstance(value, ImagePayload):
        h, w = value.array.shape[:2]
        return [_ImageBlock(
            title=f"{name}  {w}×{h}  {value.color_space}",
            array=_image_to_rgb(value),
        )]

    if isinstance(value, TensorPayload):
        return [_ImageBlock(
            title=f"{name}  {value.array.shape}  {value.dtype}  ·  ch0 heatmap",
            array=_tensor_to_heatmap(value),
        )]

    if isinstance(value, ResizeTransform):
        return [_TextBlock(name, [
            ("scale",    _fmt_floats(value.scale)),
            ("pad",      _fmt_floats(value.pad)),
            ("original", str(value.original_shape)),
            ("resized",  str(value.resized_shape)),
        ])]

    if isinstance(value, RuntimeOutputs):
        return [_TextBlock(name, [(n, str(t.array.shape)) for n, t in zip(value.names, value.tensors)])]

    if isinstance(value, TensorRegistry):
        return [_TextBlock(name, [(k, str(v.shape)) for k, v in value._tensors.items()])]

    if isinstance(value, Segmentations):
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}  mask✓")
                for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [_TextBlock(f"{name} ({n})", rows)]

    if isinstance(value, Detections):
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}")
                for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return [_TextBlock(f"{name} ({n})", rows)]

    if isinstance(value, bytes):
        return [_TextBlock(name, [("size", f"{len(value) / 1024:.1f} KB")])]

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [_TextBlock(name, [(f.name, str(getattr(value, f.name))) for f in dataclasses.fields(value)])]

    text = repr(value)
    return [_TextBlock(name, [("", text[:120] + ("…" if len(text) > 120 else ""))])]


def _to_step_view(span: StepSpan, last_image: np.ndarray | None) -> tuple[_StepView, np.ndarray | None]:
    """Convert a StepSpan to a _StepView and return the updated last_image."""
    if span.error:
        return _StepView(span.label, span.operator_config, [], error=True), last_image

    val = span.output_value
    if val is None:
        return _StepView(span.label, span.operator_config, []), last_image

    raw_blocks = _value_to_blocks(val)

    # Find first image block to update last_image
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
        # Carry forward: prepend a dimmed image block, keep text blocks
        carry = _ImageBlock(title="↑ previous", array=last_image, dim=True)
        blocks = [carry] + raw_blocks
    else:
        blocks = raw_blocks

    return _StepView(span.label, span.operator_config, blocks), last_image


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
    """Renders a list of _StepView objects as an HTML card strip."""

    def render(self, views: list[_StepView]) -> str:
        cards = [self._card(v) for v in views]
        return f'{_CSS}<div class="insp-container">{"".join(cards)}</div>'

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
    """Renders a list of _StepView objects as a matplotlib Figure."""

    def __init__(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> None:
        self.cols = cols
        self.cell_w = cell_w
        self.cell_h = cell_h

    def render(self, views: list[_StepView]) -> "matplotlib.figure.Figure":
        import matplotlib
        import matplotlib.pyplot as plt

        n = len(views)
        rows = max(1, (n + self.cols - 1) // self.cols)
        fig, axes = plt.subplots(rows, self.cols,
                                 figsize=(self.cols * self.cell_w, rows * self.cell_h))
        ax_flat: list[matplotlib.axes.Axes] = np.array(axes).flatten().tolist()

        for i, view in enumerate(views):
            self._axes(ax_flat[i], view)
        for ax in ax_flat[n:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.5)
        return fig

    def _axes(self, ax: "matplotlib.axes.Axes", view: _StepView) -> None:
        ax.set_xticks([])
        ax.set_yticks([])

        if view.error:
            for spine in ax.spines.values():
                spine.set_edgecolor("#c00")
                spine.set_linewidth(2)
            ax.set_title(view.label, fontsize=7.5, fontweight="bold", pad=3, loc="left")
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
            view.label + cfg_short,
            fontsize=7.5, fontweight="bold", pad=3, loc="left",
        )


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------

class InspectionResult:
    """The result of Pipeline.inspect(): one entry per executed step."""

    def __init__(self, spans: list[StepSpan]) -> None:
        self.spans = spans

    # -- views (lazy, shared by both renderers) --

    def _build_views(self) -> list[_StepView]:
        views: list[_StepView] = []
        last_image: np.ndarray | None = None
        for span in self.spans:
            view, last_image = _to_step_view(span, last_image)
            views.append(view)
        return views

    # -- text repr --

    def __repr__(self) -> str:
        lines = ["InspectionResult:"]
        for span in self.spans:
            shape = span.output_shape or ""
            err = " [ERROR]" if span.error else ""
            lines.append(f"  {span.label:35s}  {str(shape):20s}{err}")
        return "\n".join(lines)

    # -- Jupyter HTML --

    def _repr_html_(self) -> str:
        return HtmlRenderer().render(self._build_views())

    # -- matplotlib --

    def plot(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> "matplotlib.figure.Figure":
        """Render each step as a matplotlib subplot. Returns the Figure."""
        return PlotRenderer(cols=cols, cell_w=cell_w, cell_h=cell_h).render(self._build_views())

    # -- save to file --

    def save(self, path: str | Path | None = None, open_browser: bool = True) -> Path:
        """Write the HTML report to a file. Prints the path. Opens browser unless open_browser=False."""
        if path is None:
            fd, tmp = tempfile.mkstemp(suffix=".html", prefix="ml_pipes_inspect_")
            os.close(fd)
            out = Path(tmp)
        else:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)

        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{self._repr_html_()}</body></html>",
            encoding="utf-8",
        )
        print(f"Inspection report saved to: {out}")
        if open_browser:
            webbrowser.open(out.as_uri())
        return out
