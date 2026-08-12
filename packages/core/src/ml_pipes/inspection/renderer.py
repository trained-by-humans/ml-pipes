from __future__ import annotations

import base64
import html as _html
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Literal, Protocol, cast
import webbrowser

from ml_pipes.inspection._deps import load_cv2
from ml_pipes.inspection.views import (
    GroupBlock,
    ImageBlock,
    OutputBlock,
    StepView,
    TextBlock,
)

cv2 = load_cv2()
_IN_JUPYTER: bool = "get_ipython" in dir(__builtins__) if isinstance(__builtins__, dict) else hasattr(__builtins__, "get_ipython")
Orientation = Literal["horizontal", "vertical"]
_ORIENTATIONS: tuple[Orientation, ...] = ("horizontal", "vertical")
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


class Renderer(Protocol):
    """Anything that can turn a list of StepViews into an output format."""

    def render(
        self,
        views: list[StepView],
        orientation: Orientation = "horizontal",
    ) -> Any: ...


def _normalize_orientation(orientation: str) -> Orientation:
    normalized = orientation.strip().lower()
    if normalized not in _ORIENTATIONS:
        raise ValueError(
            f"Invalid inspection orientation: {orientation!r}. "
            f"Expected one of {list(_ORIENTATIONS)}."
        )
    return cast(Orientation, normalized)


def _flatten_step_views(views: list[StepView], depth: int = 0) -> list[tuple[StepView, int]]:
    """Pre-order traversal of a StepView tree, each entry paired with its depth."""

    flat = []
    for view in views:
        flat.append((view, depth))
        flat.extend(_flatten_step_views(view.children, depth + 1))
    return flat


class HtmlRenderer:
    """Renders a list of StepViews as HTML cards.

    Example::

        views = PipelineInspector().build_views(result)
        html: str = HtmlRenderer().render(views)
        HtmlRenderer().save(views, "report.html")
    """

    def render(
        self,
        views: list[StepView],
        orientation: Orientation = "horizontal",
    ) -> str:
        """Return a self-contained HTML string."""
        normalized_orientation = _normalize_orientation(orientation)
        cards = [self._render_card(view) for view, _ in _flatten_step_views(views)]
        return (
            f'{_CSS}<div class="insp-container insp-container--{normalized_orientation}">'
            f'{"".join(cards)}</div>'
        )

    def save(
        self,
        views: list[StepView],
        path: str | Path,
        orientation: Orientation = "horizontal",
    ) -> Path:
        """Write the HTML report to *path* and return it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Pipeline inspection</title></head><body>"
            f"{self.render(views, orientation=orientation)}</body></html>",
            encoding="utf-8",
        )
        return out

    def show(
        self,
        views: list[StepView],
        orientation: Orientation = "horizontal",
    ) -> None:
        """Display the HTML report inline in Jupyter or open it in a browser."""

        if _IN_JUPYTER:
            from IPython.display import HTML, display

            display(HTML(self.render(views, orientation=orientation)))
            return

        fd, tmp = tempfile.mkstemp(suffix=".html", prefix="ml_pipes_inspect_")
        os.close(fd)
        out = self.save(views, tmp, orientation=orientation)
        uri = out.as_uri()
        print(f"Inspection report saved to: {out}", file=sys.stderr)
        print("Opening inspection report in browser...", file=sys.stderr)
        opened = webbrowser.open(uri)
        if opened is False:
            print(
                "Browser launch was not confirmed. If nothing opened, use the saved report path above.",
                file=sys.stderr,
            )

    def _render_card(self, view: StepView) -> str:
        error_cls = " insp-card-error" if view.error else ""
        tooltip = self._render_config_tooltip(view.operator_config)
        body = self._render_body(view)
        return (
            f'<div class="insp-card{error_cls}">'
            f'<div class="insp-card-head">'
            f'<div class="insp-card-name-wrap">'
            f'<span class="insp-card-name">{_html.escape(view.label)}</span>'
            f"{tooltip}"
            f"</div></div>"
            f'<div class="insp-card-body">{body}</div>'
            f"</div>"
        )

    def _render_body(self, view: StepView) -> str:
        if view.error:
            return '<div style="font-size:12px;color:#c00;padding:4px 0;">Error during execution</div>'
        if not view.blocks:
            return "<em style='font-size:11px;color:#aaa;'>no value captured</em>"
        return "".join(self._render_block(block) for block in view.blocks)

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
                _, overlay_buf = cv2.imencode(".png", cv2.cvtColor(block.overlay_array, cv2.COLOR_RGB2BGR))
                overlay_uri = "data:image/png;base64," + base64.b64encode(overlay_buf).decode()
                toggle_style = (
                    f"{_IMG_STYLE}opacity:{opacity};cursor:pointer;"
                    "border:2px dashed #7aaef5;border-radius:3px;"
                    "transition:border-color .15s;"
                    "box-sizing:border-box;"
                )
                img_html = (
                    f'<img src="{uri}" data-primary="{uri}" data-overlay="{overlay_uri}"'
                    f' style="{toggle_style}"'
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
            dim = block.dim
            dim_class = " insp-group--dim" if dim else ""
            title_style = _TITLE_STYLE + ("color:#bbb;" if dim else "color:#555;")
            title_html = f'<div style="{title_style}">{_html.escape(block.title)}</div>'
            children = (
                self._render_group_children(block.children)
                if block.children
                else '<div class="insp-group-empty">empty</div>'
            )
            return (
                f'<div class="insp-group{dim_class}">'
                f"{title_html}"
                f'<div class="insp-group-body">{children}</div>'
                f"</div>"
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
            f'<tr><td style="{_TD_K}">{_html.escape(key)}</td>'
            f'<td style="{_TD_V}">{_html.escape(value)}</td></tr>'
            for key, value in block.rows
        )
        table = f'<table style="{_TBL_STYLE}">{inner}</table>'
        return f'<div style="margin-bottom:8px;">{title_html}{table}</div>'

    def _render_config_tooltip(self, cfg: dict) -> str:
        if not cfg:
            return ""
        rows = "".join(
            f'<tr><td class="insp-cfg-key">{_html.escape(key)}</td>'
            f'<td class="insp-cfg-val">{_html.escape(repr(value))}</td></tr>'
            for key, value in cfg.items()
        )
        table_html = f"<table>{rows}</table>"
        return f'<span class="insp-cfg-icon" data-cfg="{_html.escape(table_html, quote=True)}">⚙</span>'
