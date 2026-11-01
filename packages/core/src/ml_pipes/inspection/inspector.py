"""Pipeline inspection orchestration and view construction."""

from __future__ import annotations

import dataclasses
import warnings
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
)

ValueT = TypeVar("ValueT")

if TYPE_CHECKING:
    from ml_pipes.inspection.html_renderer import HtmlRenderer


def _is_primitive_tuple(value: tuple[Any, ...]) -> bool:
    primitive_types = (bool, int, float, str, bytes, type(None), np.generic)
    return bool(value) and all(isinstance(item, primitive_types) for item in value)


class InspectionWarning(UserWarning):
    """Warning emitted when inspection must degrade its value rendering."""


def _pydantic_base_model_type(value: Any) -> type[Any] | None:
    for base in type(value).__mro__:
        if base.__name__ == "BaseModel" and base.__module__.startswith("pydantic"):
            return base
    return None


class PipelineInspector:
    """Converts an InspectionResult into views and renders them."""

    def __init__(self) -> None:
        from ml_pipes.inspection._builtin_formatters import ensure_builtin_formatters_registered
        from ml_pipes.inspection.html_renderer import HtmlRenderer

        ensure_builtin_formatters_registered()
        self._renderer: HtmlRenderer = HtmlRenderer()
        self._formatters = FormatterRegistry(parent=global_formatter_registry())
        self._warned_value_types: set[type[Any]] = set()

    def register_value_formatter(
        self,
        value_type: type[ValueT],
        formatter: ValueFormatter[ValueT],
        *,
        override: bool = False,
    ) -> "PipelineInspector":
        """Register a formatter for inspected values of *value_type*. Returns self for chaining."""

        self._formatters.register_value_formatter(
            value_type,
            formatter,
            override=override,
        )
        return self

    def register_step_formatter(
        self,
        operator_type: type[Any],
        formatter: StepFormatter,
        *,
        override: bool = False,
    ) -> "PipelineInspector":
        """Register a formatter for inspected steps of *operator_type*. Returns self for chaining."""

        self._formatters.register_step_formatter(
            operator_type,
            formatter,
            override=override,
        )
        return self

    def _warn_for_value(self, value: Any, message: str) -> None:
        value_type = type(value)
        if value_type in self._warned_value_types:
            return
        warnings.warn(message, InspectionWarning, stacklevel=3)
        self._warned_value_types.add(value_type)

    def build_views(self, result: InspectionResult) -> list[StepView]:
        """Prepare complete formatted blocks without display limits for any renderer."""

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

        value = span.output_value
        if isinstance(value, tuple) and not _is_primitive_tuple(value):
            # A step's outer tuple is a pipeline bundle; nested tuples retain
            # their collection boundaries.
            active_ids = {id(value)}
            raw_blocks = [
                block
                for item in value
                for block in self._value_to_blocks(item, active_ids)
            ]
        else:
            raw_blocks = self._value_to_blocks(value)
        blocks, image_to_carry = _apply_image_carry(raw_blocks, last_image)
        children, _ = self._trace_to_views(span.child_trace, image_to_carry)
        return StepView(span.label, _build_span_metadata(span), blocks, children=children), image_to_carry

    def _value_to_blocks(
        self,
        value: Any,
        active_ids: set[int] | None = None,
    ) -> list[OutputBlock]:
        if active_ids is None:
            active_ids = set()

        if isinstance(value, tuple) and _is_primitive_tuple(value):
            return [TextBlock(type(value).__name__, [("", str(value))])]

        if isinstance(value, tuple) or (isinstance(value, list) and value):
            value_id = id(value)
            if value_id in active_ids:
                return [
                    GroupBlock(
                        title=self._sequence_title(value),
                        children=[self._recursive_reference_block(value)],
                    )
                ]
            active_ids.add(value_id)
            try:
                return [self._sequence_to_group(value, active_ids)]
            finally:
                active_ids.remove(value_id)

        pydantic_base_model = _pydantic_base_model_type(value)
        if (
            pydantic_base_model is not None
            and self._formatters._find_exact_value_formatter(pydantic_base_model) is None
        ):
            value_type = type(value)
            self._warn_for_value(
                value,
                f"Pydantic is unavailable, so inspection cannot structurally render "
                f"{value_type.__module__}.{value_type.__qualname__}; using repr instead.",
            )
            text = value if isinstance(value, str) else repr(value)
            return [TextBlock(type(value).__name__, [("", text)])]

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
        return [TextBlock(name, [("", text)])]

    @staticmethod
    def _sequence_title(value: list[Any] | tuple[Any, ...]) -> str:
        if isinstance(value, tuple):
            return f"tuple  ×{len(value)}"
        return f"list[{type(value[0]).__name__}]  ×{len(value)}"

    def _sequence_to_group(
        self,
        value: list[Any] | tuple[Any, ...],
        active_ids: set[int],
    ) -> GroupBlock:
        return GroupBlock(
            title=self._sequence_title(value),
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
