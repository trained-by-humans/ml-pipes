from __future__ import annotations

import base64
import dataclasses
import html
import os
import tempfile
import webbrowser
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


class _CaptureCollector(TraceCollector):
    def __init__(self) -> None:
        self.trace: InvocationTrace | None = None

    def on_trace(self, trace: InvocationTrace) -> None:
        self.trace = trace


# ---------------------------------------------------------------------------
# Array → data URI
# ---------------------------------------------------------------------------

def _array_to_data_uri(arr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", arr)
    b64 = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{b64}"


def _image_payload_to_uri(value: ImagePayload) -> str:
    arr = value.array
    if value.color_space == "BGR":
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return _array_to_data_uri(arr)


def _tensor_payload_to_heatmap_uri(value: TensorPayload) -> str:
    arr = value.array
    if arr.ndim == 3:
        channel = arr[0] if value.layout.startswith("C") else arr[:, :, 0]
    elif arr.ndim == 4:
        channel = arr[0, 0] if value.layout.startswith("N") else arr[0, :, :, 0]
    else:
        channel = arr
    channel = channel.astype(np.float32)
    mn, mx = channel.min(), channel.max()
    if mx > mn:
        channel = ((channel - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        channel = np.zeros_like(channel, dtype=np.uint8)
    heat = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
    return _array_to_data_uri(heat)


# ---------------------------------------------------------------------------
# Structured text rendering (key/value table, no repr() strings)
# ---------------------------------------------------------------------------

_IMG_STYLE = "max-width:240px;max-height:200px;object-fit:contain;display:block;"
_TBL_STYLE = "font-size:11px;border-collapse:collapse;width:100%;"
_TD_K = "padding:1px 6px 1px 0;color:#555;white-space:nowrap;vertical-align:top;"
_TD_V = "padding:1px 0;word-break:break-all;vertical-align:top;"

_BLOCK_TITLE = (
    "font-size:10px;font-weight:600;color:#555;"
    "text-transform:uppercase;letter-spacing:0.04em;margin-bottom:3px;"
)
_BLOCK_TITLE_DIM = (
    "font-size:10px;font-weight:600;color:#bbb;"
    "text-transform:uppercase;letter-spacing:0.04em;margin-bottom:3px;"
)


def _kv_table(rows: list[tuple[str, str]]) -> str:
    inner = "".join(
        f'<tr><td style="{_TD_K}">{html.escape(k)}</td>'
        f'<td style="{_TD_V}">{html.escape(v)}</td></tr>'
        for k, v in rows
    )
    return f'<table style="{_TBL_STYLE}">{inner}</table>'


def _fmt_floats(seq: Any, precision: int = 3) -> str:
    try:
        return "(" + ", ".join(f"{x:.{precision}g}" for x in seq) + ")"
    except Exception:
        return str(seq)


def _named_block(title: str, content: str, dim: bool = False) -> str:
    style = _BLOCK_TITLE_DIM if dim else _BLOCK_TITLE
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="{style}">{html.escape(title)}</div>'
        f'{content}'
        f'</div>'
    )


def _render_text_content(value: Any) -> str:
    """Render a single non-image value as a titled key/value block."""
    name = type(value).__name__

    if isinstance(value, ResizeTransform):
        rows = [
            ("scale", _fmt_floats(value.scale)),
            ("pad", _fmt_floats(value.pad)),
            ("original", str(value.original_shape)),
            ("resized", str(value.resized_shape)),
        ]
        return _named_block(name, _kv_table(rows))

    if isinstance(value, RuntimeOutputs):
        rows = [(n, str(t.array.shape)) for n, t in zip(value.names, value.tensors)]
        return _named_block(name, _kv_table(rows))

    if isinstance(value, TensorRegistry):
        rows = [(k, str(v.shape)) for k, v in value._tensors.items()]
        return _named_block(name, _kv_table(rows))

    if isinstance(value, Segmentations):
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}  mask✓")
                for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return _named_block(f"{name} ({n})", _kv_table(rows))

    if isinstance(value, Detections):
        n = len(value.boxes)
        rows = [(f"[{i}]", f"cls={value.classes[i]}  score={value.scores[i]:.2f}")
                for i in range(min(n, 6))]
        if n > 6:
            rows.append(("…", f"+{n - 6} more"))
        return _named_block(f"{name} ({n})", _kv_table(rows))

    if isinstance(value, bytes):
        return _named_block(name, _kv_table([("size", f"{len(value) / 1024:.1f} KB")]))

    # Generic dataclass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        rows = [(f.name, str(getattr(value, f.name))) for f in dataclasses.fields(value)]
        return _named_block(name, _kv_table(rows))

    text = html.escape(repr(value))
    if len(text) > 300:
        text = text[:300] + "…"
    return _named_block(name, f'<pre style="font-size:11px;margin:0;white-space:pre-wrap;">{text}</pre>')


# ---------------------------------------------------------------------------
# Per-span rendering
# ---------------------------------------------------------------------------

def _find_image_element(value: Any) -> ImagePayload | TensorPayload | None:
    if isinstance(value, (ImagePayload, TensorPayload)):
        return value
    if isinstance(value, tuple):
        for item in value:
            found = _find_image_element(item)
            if found is not None:
                return found
    return None


def _non_image_elements(value: Any) -> list[Any]:
    if isinstance(value, (ImagePayload, TensorPayload)):
        return []
    if isinstance(value, tuple):
        return [item for item in value if not isinstance(item, (ImagePayload, TensorPayload))]
    return [value]


def _render_span_body(value: Any, last_image_uri: str | None) -> tuple[str, str | None]:
    """Return (html_body, new_last_image_uri).

    Each output element is rendered as a named block (title + content), stacked
    vertically. Image/tensor blocks appear first; non-image blocks below.
    Carry-forward images use a dimmed title so it's clear no new image was produced.
    """
    img_val = _find_image_element(value)
    new_image = img_val is not None

    parts: list[str] = []

    if new_image:
        last_image_uri = (
            _image_payload_to_uri(img_val)
            if isinstance(img_val, ImagePayload)
            else _tensor_payload_to_heatmap_uri(img_val)
        )
        if isinstance(img_val, ImagePayload):
            h, w = img_val.array.shape[:2]
            title = f"ImagePayload  {w}×{h}  {img_val.color_space}"
            img_html = f'<img src="{last_image_uri}" style="{_IMG_STYLE}">'
        else:
            title = f"TensorPayload  {img_val.array.shape}  {img_val.dtype}"
            note = html.escape(f"layout: {img_val.layout}  · channel 0 heatmap")
            img_html = (
                f'<img src="{last_image_uri}" style="{_IMG_STYLE}">'
                f'<div style="font-size:10px;color:#888;margin-top:2px;">{note}</div>'
            )
        parts.append(_named_block(title, img_html))

    elif last_image_uri:
        img_html = f'<img src="{last_image_uri}" style="{_IMG_STYLE}opacity:0.2;">'
        parts.append(_named_block("↑ previous", img_html, dim=True))

    for item in _non_image_elements(value):
        parts.append(_render_text_content(item))

    body = "\n".join(parts) if parts else "<em style='font-size:11px;color:#aaa;'>no output</em>"
    return body, last_image_uri


# ---------------------------------------------------------------------------
# CSS + JS
# ---------------------------------------------------------------------------

_CARD_CSS = """
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
.insp-card-name-wrap {
  display: flex;
  align-items: center;
  gap: 5px;
}
.insp-card-name {
  font-size: 12px;
  font-weight: 600;
  color: #222;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
}
.insp-cfg-icon {
  font-size: 10px;
  color: #999;
  cursor: default;
  line-height: 1;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 0 3px;
  background: #fff;
  user-select: none;
  flex-shrink: 0;
}
.insp-cfg-icon:hover { color: #333; border-color: #888; }
.insp-card-body {
  padding: 7px 8px;
  border-radius: 0 0 6px 6px;
}
.insp-card-error .insp-card-head {
  background: #fff0f0;
  border-color: #f5a0a0;
}
.insp-card-error .insp-card-name { color: #c00; }

/* Shared fixed tooltip popup */
#insp-cfg-popup {
  display: none;
  position: fixed;
  z-index: 9999;
  background: #1e1e1e;
  color: #d4d4d4;
  border-radius: 5px;
  padding: 8px 10px;
  font-size: 11px;
  font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
  white-space: nowrap;
  box-shadow: 0 4px 14px rgba(0,0,0,.4);
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
    var r = icon.getBoundingClientRect();
    var left = r.left;
    var top = r.bottom + 6;
    popup.style.display = 'block';
    // nudge left if it would overflow viewport
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


def _config_tooltip(cfg: dict) -> str:
    """Render operator config as a ⚙ icon; tooltip is shown via fixed-position JS popup."""
    if not cfg:
        return ""
    rows = "".join(
        f'<tr><td class="insp-cfg-key">{html.escape(k)}</td>'
        f'<td class="insp-cfg-val">{html.escape(repr(v))}</td></tr>'
        for k, v in cfg.items()
    )
    table_html = f"<table>{rows}</table>"
    attr_val = html.escape(table_html, quote=True)
    return f'<span class="insp-cfg-icon" data-cfg="{attr_val}">⚙</span>'


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------

class InspectionResult:
    """The result of Pipeline.inspect(): one entry per executed step."""

    def __init__(self, spans: list[StepSpan]) -> None:
        self.spans = spans

    def __repr__(self) -> str:
        lines = ["InspectionResult:"]
        for span in self.spans:
            shape = span.output_shape or ""
            err = " [ERROR]" if span.error else ""
            lines.append(f"  {span.label:35s}  {str(shape):20s}{err}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        cards = []
        last_image_uri: str | None = None

        for span in self.spans:
            error_cls = " insp-card-error" if span.error else ""

            if span.error:
                body_html = '<div style="font-size:12px;color:#c00;padding:4px 0;">Error during execution</div>'
            else:
                val = span.output_value
                if val is not None:
                    body_html, last_image_uri = _render_span_body(val, last_image_uri)
                else:
                    body_html = "<em style='font-size:11px;color:#aaa;'>no value captured</em>"

            tooltip = _config_tooltip(span.operator_config)
            cards.append(
                f'<div class="insp-card{error_cls}">'
                f'  <div class="insp-card-head">'
                f'    <div class="insp-card-name-wrap">'
                f'      <span class="insp-card-name">{html.escape(span.label)}</span>'
                f'      {tooltip}'
                f'    </div>'
                f'  </div>'
                f'  <div class="insp-card-body">{body_html}</div>'
                f'</div>'
            )

        cards_html = "\n".join(cards)
        return f'{_CARD_CSS}<div class="insp-container">{cards_html}</div>'

    def save(self, path: str | Path | None = None, open_browser: bool = True) -> Path:
        """Write the inspection report to an HTML file and return its path.

        If *path* is None a temporary file is created. Prints the path to stdout.
        Pass open_browser=False to suppress auto-opening.
        """
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
