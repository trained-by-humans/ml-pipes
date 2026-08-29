from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

import numpy as np

from ml_pipes.inspection._global_registry import (
    register_step_formatter,
    register_value_formatter,
)
from ml_pipes.inspection.registry import ValueFormatter
from ml_pipes.region import RegionOpener
from ml_pipes.tracing import StepSpan, _fmt_batch_size
from ml_pipes.inspection.views import (
    GroupBlock,
    ImageBlock,
    OutputBlock,
    StepView,
    TextBlock,
    _build_span_metadata,
)


def _is_rgb_image_array(value: np.ndarray) -> bool:
    return value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3


def ndarray_image_formatter(
    *,
    default_color_space: Literal["RGB", "BGR"] = "RGB",
) -> ValueFormatter[np.ndarray]:
    """Create an ndarray formatter that previews HWC uint8 images in the given color space.

    Bare ndarrays do not carry color-space metadata. Use this factory when an
    application knows the convention its image arrays follow.
    """

    if default_color_space not in {"RGB", "BGR"}:
        raise ValueError("default_color_space must be 'RGB' or 'BGR'")

    def format_ndarray(value: np.ndarray) -> list[OutputBlock]:
        if _is_rgb_image_array(value):
            height, width = value.shape[:2]
            image = value if default_color_space == "RGB" else np.ascontiguousarray(value[:, :, ::-1])
            return [
                ImageBlock(title=f"ndarray  {width}×{height}  {default_color_space}", array=image),
                TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))]),
            ]
        return [TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))])]

    return format_ndarray


def _format_ndarray(value: np.ndarray) -> list[OutputBlock]:
    return ndarray_image_formatter()(value)


def _format_bytes(value: bytes) -> list[OutputBlock]:
    return [TextBlock("bytes", [("size", f"{len(value) / 1024:.1f} KB")])]


@dataclass
class _PydanticRenderState:
    max_nodes: int | None
    active_ids: set[int] = field(default_factory=set)
    node_count: int = 0

    def can_add_node(self) -> bool:
        return self.max_nodes is None or self.node_count < self.max_nodes

    def add_node(self) -> None:
        self.node_count += 1


def _validate_pydantic_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")


def _pydantic_fields(value: Any) -> Mapping[Any, Any] | None:
    fields = getattr(type(value), "model_fields", None)
    if isinstance(fields, Mapping):
        return fields

    fields = getattr(type(value), "__fields__", None)
    if isinstance(fields, Mapping):
        return fields
    return None


def pydantic_model_formatter(
    *,
    max_depth: int | None = None,
    max_members: int | None = None,
    max_items: int | None = None,
    max_text_length: int | None = None,
    max_nodes: int | None = None,
) -> ValueFormatter[Any]:
    """Create a structural formatter for Pydantic v1 and v2 models.

    Each limit is optional. ``None`` keeps the corresponding part of the
    captured model unbounded; recursive references are still detected.
    """

    for name, value in {
        "max_depth": max_depth,
        "max_members": max_members,
        "max_items": max_items,
        "max_text_length": max_text_length,
        "max_nodes": max_nodes,
    }.items():
        _validate_pydantic_limit(name, value)

    def limit_text(text: str) -> str:
        if max_text_length is None or len(text) <= max_text_length:
            return text
        return text[:max_text_length] + "…"

    def text_for(value: Any) -> str:
        if isinstance(value, str):
            return limit_text(value)
        try:
            return limit_text(repr(value))
        except Exception as exc:  # pragma: no cover - hostile third-party repr
            return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"

    def limit_block(message: str) -> TextBlock:
        return TextBlock("…", [("", message)])

    def render_member(name: str, value: Any, state: _PydanticRenderState, depth: int) -> OutputBlock:
        blocks = render_value(value, state, depth)
        if len(blocks) == 1:
            block = blocks[0]
            if isinstance(block, GroupBlock):
                return GroupBlock(title=f"{name}: {block.title}", children=block.children, dim=block.dim)
            if isinstance(block, TextBlock) and len(block.rows) == 1 and block.rows[0][0] == "":
                return TextBlock("", [(name, block.rows[0][1])], dim=block.dim)
        return GroupBlock(title=f"{name}: {type(value).__name__}", children=blocks)

    def render_members(
        members: Iterable[tuple[str, Any]],
        member_count: int,
        state: _PydanticRenderState,
        depth: int,
        member_limit: int | None,
        *,
        wrap_items: bool = False,
    ) -> list[OutputBlock]:
        children: list[OutputBlock] = []
        shown = 0
        for name, item in members:
            if member_limit is not None and shown >= member_limit:
                children.append(limit_block(f"+{member_count - shown} more"))
                break
            if not state.can_add_node():
                children.append(limit_block("node limit reached"))
                break
            if wrap_items:
                state.add_node()
                children.append(GroupBlock(title=name, children=render_value(item, state, depth)))
            else:
                children.append(render_member(name, item, state, depth))
            shown += 1
        return children

    def render_group(
        title: str,
        members: Iterable[tuple[str, Any]],
        member_count: int,
        state: _PydanticRenderState,
        depth: int,
        member_limit: int | None,
        value: Any,
        *,
        wrap_items: bool = False,
    ) -> list[OutputBlock]:
        if max_depth is not None and depth >= max_depth:
            return [limit_block("maximum depth reached")]
        if id(value) in state.active_ids:
            return [TextBlock(type(value).__name__, [("", f"<recursive {type(value).__name__}>")])]
        if not state.can_add_node():
            return [limit_block("node limit reached")]

        state.add_node()
        state.active_ids.add(id(value))
        try:
            return [
                GroupBlock(
                    title=title,
                    children=render_members(
                        members,
                        member_count,
                        state,
                        depth + 1,
                        member_limit,
                        wrap_items=wrap_items,
                    ),
                )
            ]
        finally:
            state.active_ids.remove(id(value))

    def render_value(value: Any, state: _PydanticRenderState, depth: int) -> list[OutputBlock]:
        fields = _pydantic_fields(value)
        if fields is not None:
            def model_members() -> Iterable[tuple[str, Any]]:
                for name in fields:
                    try:
                        yield str(name), getattr(value, name)
                    except Exception as exc:  # pragma: no cover - defensive third-party model access
                        yield str(name), f"<unavailable: {type(exc).__name__}>"

            return render_group(
                type(value).__name__,
                model_members(),
                len(fields),
                state,
                depth,
                max_members,
                value,
            )

        if isinstance(value, Mapping):
            return render_group(
                type(value).__name__,
                ((str(key), item) for key, item in value.items()),
                len(value),
                state,
                depth,
                max_members,
                value,
            )

        if isinstance(value, (list, tuple)):
            item_type = type(value[0]).__name__ if value else "Any"
            return render_group(
                f"{type(value).__name__}[{item_type}]  ×{len(value)}",
                ((f"[{index}]", item) for index, item in enumerate(value)),
                len(value),
                state,
                depth,
                max_items,
                value,
                wrap_items=True,
            )

        if not state.can_add_node():
            return [limit_block("node limit reached")]
        state.add_node()
        return [TextBlock(type(value).__name__, [("", text_for(value))])]

    def format_pydantic_model(value: Any) -> list[OutputBlock]:
        return render_value(value, _PydanticRenderState(max_nodes=max_nodes), 0)

    return format_pydantic_model


def _register_pydantic_base_model_formatter() -> None:
    try:
        from pydantic import BaseModel
    except ImportError:
        return

    register_value_formatter(BaseModel, pydantic_model_formatter())


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


def _region_step_formatter(
    span: StepSpan,
    last_image: np.ndarray | None,
) -> tuple[StepView, np.ndarray | None]:
    return StepView(span.label, _build_span_metadata(span), _region_summary_block(span)), last_image


_BUILTINS_LOCK = Lock()
_BUILTINS_REGISTERED = False


def ensure_builtin_formatters_registered() -> None:
    global _BUILTINS_REGISTERED

    if _BUILTINS_REGISTERED:
        return
    with _BUILTINS_LOCK:
        if _BUILTINS_REGISTERED:
            return
        register_value_formatter(np.ndarray, _format_ndarray)
        register_value_formatter(bytes, _format_bytes)
        register_step_formatter(RegionOpener, _region_step_formatter)
        _register_pydantic_base_model_formatter()
        _BUILTINS_REGISTERED = True
