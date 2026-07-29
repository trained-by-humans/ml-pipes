"""Pipeline inspection orchestration and view construction."""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipes.inspection.artifacts import InspectionResult
from ml_pipes.inspection.views import (
    GroupBlock,
    ImageBlock,
    OutputBlock,
    OutputFormatter,
    Renderer,
    SpanFormatter,
    StepView,
    TextBlock,
    _apply_image_carry,
    _build_span_metadata,
    _make_grid,
)

_IN_JUPYTER: bool = "get_ipython" in dir(__builtins__) if isinstance(__builtins__, dict) else hasattr(__builtins__, "get_ipython")


def _block_summary(blocks: list[OutputBlock]) -> str:
    """Collapse a block list to a single short string for list-item rows."""

    parts = []
    for block in blocks:
        if isinstance(block, ImageBlock):
            parts.append(block.title)
        elif isinstance(block, TextBlock):
            summary = block.title
            rows = block.rows[:3] if block.title == "dict" else block.rows[:1]
            if rows:
                row_summaries = [(f"{key} {value}".rstrip() if key else f"{value}") for key, value in rows]
                if summary:
                    summary += "  " + "  |  ".join(row_summaries)
                else:
                    summary = "  |  ".join(row_summaries)
            parts.append(summary)
        else:
            summary = block.title
            child_summaries = [_block_summary([child]) for child in block.children[:2]]
            if child_summaries:
                summary += "  " + "  |  ".join(child_summaries)
            if len(block.children) > 2:
                summary += f"  |  +{len(block.children) - 2} more"
            parts.append(summary)
    return "  |  ".join(parts)


def _is_primitive_tuple(value: tuple[Any, ...]) -> bool:
    primitive_types = (bool, int, float, str, bytes, type(None), np.generic)
    return bool(value) and all(isinstance(item, primitive_types) for item in value)


class PipelineInspector:
    """Converts an InspectionResult into views and renders them."""

    def __init__(self) -> None:
        from ml_pipes.inspection.formatters import default_output_formatters, default_span_formatters

        self._output_fmts: dict[type, OutputFormatter] = default_output_formatters()
        self._span_fmts: dict[type, SpanFormatter] = default_span_formatters()

    def register_output_formatter(self, type_: type, formatter: OutputFormatter) -> "PipelineInspector":
        """Register a formatter for *type_* output values. Returns self for chaining."""

        self._output_fmts[type_] = formatter
        return self

    def register_span_formatter(self, operator_type: type, formatter: SpanFormatter) -> "PipelineInspector":
        """Register a span-level formatter for *operator_type*. Returns self for chaining."""

        self._span_fmts[operator_type] = formatter
        return self

    def _find_output_formatter(self, value: Any) -> OutputFormatter | None:
        value_type = type(value)
        return self._output_fmts.get(value_type) or next(
            (
                formatter
                for registered_type, formatter in self._output_fmts.items()
                if issubclass(value_type, registered_type)
            ),
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
            list_max = 6
            formatter = self._find_output_formatter(value[0])
            if formatter is not None:
                all_blocks = [formatter(item) for item in value]
                first = all_blocks[0]
                if first and isinstance(first[0], ImageBlock):
                    grid = _make_grid(
                        [block.array for blocks in all_blocks for block in blocks if isinstance(block, ImageBlock)],
                        divider=2,
                    )
                    return [ImageBlock(title=f"{first[0].title.split('  ')[0]}  ×{len(value)}", array=grid)]
                rows = [(f"[{i}]", _block_summary(blocks)) for i, blocks in enumerate(all_blocks[:list_max])]
                if len(value) > list_max:
                    rows.append(("…", f"+{len(value) - list_max} more"))
                return [TextBlock(f"list  ×{len(value)}", rows)]
            if isinstance(value[0], Mapping) or (dataclasses.is_dataclass(value[0]) and not isinstance(value[0], type)):
                rows = [
                    (f"[{i}]", _block_summary(self._output_to_blocks(item, active_ids)))
                    for i, item in enumerate(value[:list_max])
                ]
                if len(value) > list_max:
                    rows.append(("…", f"+{len(value) - list_max} more"))
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
                next(
                    (
                        fmt
                        for registered_type, fmt in self._span_fmts.items()
                        if issubclass(op_type, registered_type)
                    ),
                    None,
                )
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

        from ml_pipes.inspection.html_renderer import HtmlRenderer

        return HtmlRenderer(orientation=orientation).render(self.build_views(result))

    def save_to_html(
        self,
        result: InspectionResult,
        path: str | Path,
        orientation: str = "horizontal",
    ) -> Path:
        """Write an HTML report to *path* and return it."""

        from ml_pipes.inspection.html_renderer import HtmlRenderer

        return HtmlRenderer(orientation=orientation).save(self.build_views(result), path)

    def to_plot(
        self,
        result: InspectionResult,
        cols: int = 6,
        cell_w: float = 2.6,
        cell_h: float = 3.2,
    ) -> "matplotlib.figure.Figure":
        """Return a matplotlib Figure."""

        from ml_pipes.inspection.plot_renderer import PlotRenderer

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
        """Display the result."""

        if _IN_JUPYTER:
            from IPython.display import HTML, display

            display(HTML(self.to_html(result, orientation=orientation)))
        else:
            self.to_plot(result, cols=cols)

            import matplotlib.pyplot as plt

            plt.show()

    def show_in_browser(self, result: InspectionResult, orientation: str = "horizontal") -> None:
        """Save the HTML report to a temp file, announce it, and open it in the default browser."""

        fd, tmp = tempfile.mkstemp(suffix=".html", prefix="ml_pipes_inspect_")
        os.close(fd)
        out = self.save_to_html(result, tmp, orientation=orientation)
        uri = out.as_uri()
        print(f"Inspection report saved to: {out}", file=sys.stderr)
        print("Opening inspection report in browser...", file=sys.stderr)
        opened = webbrowser.open(uri)
        if opened is False:
            print(
                "Browser launch was not confirmed. If nothing opened, use the saved report path above.",
                file=sys.stderr,
            )
