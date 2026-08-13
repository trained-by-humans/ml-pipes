"""Pipeline inspection orchestration and view construction."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ml_pipes.inspection.artifacts import InspectionResult
from ml_pipes.inspection._global_registry import global_formatter_registry
from ml_pipes.inspection.registry import (
    FormatterRegistry,
    StepFormatter,
    ValueFormatter,
)
from ml_pipes.inspection.renderer import _normalize_orientation
from ml_pipes.tracing import StepSpan
from ml_pipes.inspection.views import (
    GroupBlock,
    ImageBlock,
    OutputBlock,
    StepView,
    TextBlock,
    _apply_image_carry,
    _build_span_metadata,
    _make_grid,
)

ValueT = TypeVar("ValueT")

if TYPE_CHECKING:
    from ml_pipes.inspection.renderer import HtmlRenderer


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
        from ml_pipes.inspection._builtin_formatters import ensure_builtin_formatters_registered
        from ml_pipes.inspection.renderer import HtmlRenderer

        ensure_builtin_formatters_registered()
        self._renderer: HtmlRenderer = HtmlRenderer()
        self._formatters = FormatterRegistry(parent=global_formatter_registry())

    def register_value_formatter(
        self,
        value_type: type[ValueT],
        formatter: ValueFormatter[ValueT],
    ) -> "PipelineInspector":
        """Register a formatter for inspected values of *value_type*. Returns self for chaining."""

        self._formatters.register_value_formatter(value_type, formatter)
        return self

    def register_step_formatter(self, operator_type: type[Any], formatter: StepFormatter) -> "PipelineInspector":
        """Register a formatter for inspected steps of *operator_type*. Returns self for chaining."""

        self._formatters.register_step_formatter(operator_type, formatter)
        return self

    def build_views(self, result: InspectionResult) -> list[StepView]:
        """Prepare the intermediate StepView tree from spans for built-in or custom renderers."""

        views, _ = self._trace_to_views(result, None)
        return views

    def render(
        self,
        result: InspectionResult,
        orientation: str = "horizontal",
    ) -> str:
        """Return the built-in self-contained HTML report."""

        return self._renderer.render(
            self.build_views(result),
            orientation=_normalize_orientation(orientation),
        )

    def save(
        self,
        result: InspectionResult,
        path: str | Path,
        orientation: str = "horizontal",
    ) -> Path:
        """Write the built-in HTML report to *path* and return it."""

        return self._renderer.save(
            self.build_views(result),
            path,
            orientation=_normalize_orientation(orientation),
        )

    def show(
        self,
        result: InspectionResult,
        orientation: str = "horizontal",
    ) -> None:
        """Display the result inline in Jupyter or open a browser report otherwise."""

        self._renderer.show(
            self.build_views(result),
            orientation=_normalize_orientation(orientation),
        )

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

    def _span_to_view(
        self,
        span: StepSpan,
        last_image: np.ndarray | None,
    ) -> tuple[StepView, np.ndarray | None]:
        if span.error:
            children, _ = self._trace_to_views(span.child_trace, last_image)
            return StepView(span.label, _build_span_metadata(span), [], error=True, children=children), last_image

        op_type = span.operator_type
        formatter = self._formatters.find_step_formatter(op_type) if op_type is not None else None
        if formatter is not None:
            view, image_to_carry = formatter(span, last_image)
            children, _ = self._trace_to_views(span.child_trace, image_to_carry)
            return dataclasses.replace(view, children=children), image_to_carry

        raw_blocks = self._value_to_blocks(span.output_value)
        blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
        children, _ = self._trace_to_views(span.child_trace, image_to_carry)
        return StepView(span.label, _build_span_metadata(span), blocks, children=children), image_to_carry

    def _find_value_formatter(self, value: Any) -> ValueFormatter[Any] | None:
        return self._formatters.find_value_formatter(type(value))

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
        blocks = self._value_to_blocks(value, active_ids)
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

    def _value_to_blocks(
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
                blocks.extend(self._value_to_blocks(item, active_ids))
            return blocks

        if isinstance(value, list) and value:
            list_max = 6
            formatter = self._find_value_formatter(value[0])
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
                    (f"[{i}]", _block_summary(self._value_to_blocks(item, active_ids)))
                    for i, item in enumerate(value[:list_max])
                ]
                if len(value) > list_max:
                    rows.append(("…", f"+{len(value) - list_max} more"))
                return [TextBlock(f"list  ×{len(value)}", rows)]
            item_type = type(value[0]).__name__
            return [TextBlock(f"list[{item_type}]  ×{len(value)}", [("", "…")])]

        formatter = self._find_value_formatter(value)
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
