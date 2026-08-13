"""Pipeline inspection orchestration and view construction."""

from __future__ import annotations

import dataclasses
import re
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
    _summarize_blocks,
)

ValueT = TypeVar("ValueT")
_DEFAULT_GROUP_PREVIEW_LIMIT = 12
_DEFAULT_LIST_PREVIEW_LIMIT = 6
_LIST_TITLE_RE = re.compile(
    r"^(?:(?P<label>.+): )?list\[(?P<item_type>[^\]]+)\]\s+×(?P<count>\d+)$"
)

if TYPE_CHECKING:
    from ml_pipes.inspection.renderer import HtmlRenderer


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
        compacted_blocks = self._compact_blocks(raw_blocks)
        blocks, image_to_carry = _apply_image_carry(compacted_blocks, last_image)
        children, _ = self._trace_to_views(span.child_trace, image_to_carry)
        return StepView(span.label, _build_span_metadata(span), blocks, children=children), image_to_carry

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
            value_id = id(value)
            if value_id in active_ids:
                return [
                    GroupBlock(
                        title=f"list[{type(value[0]).__name__}]  ×{len(value)}",
                        children=[self._recursive_reference_block(value)],
                    )
                ]
            active_ids.add(value_id)
            try:
                return [self._list_to_group(value, active_ids)]
            finally:
                active_ids.remove(value_id)

        formatter = self._formatters.find_value_formatter(type(value))
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

    def _list_to_group(
        self,
        value: list[Any],
        active_ids: set[int],
    ) -> GroupBlock:
        return GroupBlock(
            title=f"list[{type(value[0]).__name__}]  ×{len(value)}",
            children=[
                GroupBlock(
                    title=f"[{index}]",
                    children=self._value_to_blocks(item, active_ids),
                )
                for index, item in enumerate(value)
            ],
        )

    def _mapping_to_group(
        self,
        title: str,
        value: Mapping[Any, Any],
        active_ids: set[int],
    ) -> GroupBlock:
        return GroupBlock(
            title=title,
            children=[
                self._member_block(str(key), item, active_ids)
                for key, item in value.items()
            ],
        )

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
                self._member_block(field.name, getattr(value, field.name), active_ids)
                for field in dataclasses.fields(value)
            ],
        )

    def _member_block(
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

    def _compact_blocks(
        self,
        blocks: list[OutputBlock],
        *,
        group_preview_limit: int = _DEFAULT_GROUP_PREVIEW_LIMIT,
        list_preview_limit: int = _DEFAULT_LIST_PREVIEW_LIMIT,
    ) -> list[OutputBlock]:
        def preview_group(block: GroupBlock) -> GroupBlock:
            if len(block.children) <= group_preview_limit:
                return block
            return GroupBlock(
                title=block.title,
                children=block.children[:group_preview_limit] + [
                    TextBlock("…", [("", f"+{len(block.children) - group_preview_limit} more")], dim=block.dim)
                ],
                dim=block.dim,
            )

        def compact_list_group(block: GroupBlock) -> OutputBlock:
            parts = self._list_title_parts(block.title)
            if parts is None:
                return block

            label, item_type, count = parts
            item_groups = self._item_groups(block.children)
            preview_item_groups = item_groups[:group_preview_limit]
            preview_images = [self._preview_item_image_block(item) for item in preview_item_groups]
            if preview_item_groups and all(image is not None for image in preview_images):
                images = [image for image in preview_images if image is not None]
                title = self._image_grid_title(
                    label,
                    images[0].title.split("  ")[0],
                    count,
                    shown_count=len(images),
                )
                return ImageBlock(
                    title=title,
                    array=_make_grid(
                        [image.array for image in images],
                        divider=2,
                    ),
                    dim=block.dim,
                )

            if item_groups and not all(self._is_scalar_item_group(item) for item in item_groups):
                rows = [
                    (item.title, _summarize_blocks(item.children))
                    for item in item_groups[:list_preview_limit]
                ]
                if count > list_preview_limit:
                    rows.append(("…", f"+{count - list_preview_limit} more"))
                return TextBlock(self._list_summary_title(label, count), rows, dim=block.dim)

            return TextBlock(self._list_placeholder_title(label, item_type, count), [("", "…")], dim=block.dim)

        def compact_block(block: OutputBlock) -> OutputBlock:
            if not isinstance(block, GroupBlock):
                return block

            compacted = GroupBlock(
                title=block.title,
                children=[compact_block(child) for child in block.children],
                dim=block.dim,
            )
            list_compacted = compact_list_group(compacted)
            if isinstance(list_compacted, GroupBlock):
                return preview_group(list_compacted)
            return list_compacted

        return [compact_block(block) for block in blocks]

    def _list_title_parts(self, title: str) -> tuple[str | None, str, int] | None:
        match = _LIST_TITLE_RE.fullmatch(title)
        if match is None:
            return None
        return match.group("label"), match.group("item_type"), int(match.group("count"))

    def _item_groups(self, children: list[OutputBlock]) -> list[GroupBlock]:
        groups: list[GroupBlock] = []
        for child in children:
            if isinstance(child, GroupBlock) and child.title.startswith("[") and child.title.endswith("]"):
                groups.append(child)
        return groups

    def _preview_item_image_block(self, item: GroupBlock) -> ImageBlock | None:
        if not item.children:
            return None
        first_child = item.children[0]
        if not isinstance(first_child, ImageBlock):
            return None

        metadata_title = first_child.title.split("  ")[0]
        for child in item.children[1:]:
            if not isinstance(child, TextBlock):
                return None
            if child.title not in {"", metadata_title}:
                return None

        return first_child

    def _list_summary_title(self, label: str | None, count: int) -> str:
        title = f"list  ×{count}"
        return f"{label}: {title}" if label else title

    def _list_placeholder_title(self, label: str | None, item_type: str, count: int) -> str:
        title = f"list[{item_type}]  ×{count}"
        return f"{label}: {title}" if label else title

    def _image_grid_title(
        self,
        label: str | None,
        item_title: str,
        count: int,
        *,
        shown_count: int,
    ) -> str:
        title = f"{item_title}  ×{count}"
        if shown_count < count:
            title = f"{title}  (showing {shown_count} out of {count})"
        return f"{label}: {title}" if label else title

    def _recursive_reference_block(self, value: Any) -> TextBlock:
        return TextBlock(type(value).__name__, [("", f"<recursive {type(value).__name__}>")])

    def _anonymous_single_row_text_block(self, block: OutputBlock) -> TextBlock | None:
        if (
            isinstance(block, TextBlock)
            and len(block.rows) == 1
            and block.rows[0][0] == ""
        ):
            return block
        return None

    def _is_scalar_field_block(self, block: OutputBlock, value: Any) -> bool:
        text_block = self._anonymous_single_row_text_block(block)
        return text_block is not None and text_block.title == type(value).__name__

    def _is_scalar_item_group(self, block: GroupBlock) -> bool:
        return (
            len(block.children) == 1
            and self._anonymous_single_row_text_block(block.children[0]) is not None
        )
