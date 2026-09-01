from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import Lock
from typing import Any, Literal

import numpy as np

from ml_pipes.inspection._global_registry import (
    global_formatter_registry,
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
) -> ValueFormatter[Any]:
    """Create a structural formatter for Pydantic v1 and v2 models.

    ``max_depth`` is optional. ``None`` allows arbitrary nesting; recursive
    references are still detected.
    """

    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
    ):
        raise ValueError("max_depth must be a positive integer or None")

    def text_for(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return repr(value)
        except Exception as exc:  # pragma: no cover - hostile third-party repr
            return f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"

    def limit_block(message: str) -> TextBlock:
        return TextBlock("…", [("", message)])

    def render_member(name: str, value: Any, active_ids: set[int], depth: int) -> OutputBlock:
        blocks = render_value(value, active_ids, depth)
        if len(blocks) == 1:
            block = blocks[0]
            if isinstance(block, GroupBlock):
                return GroupBlock(title=f"{name}: {block.title}", children=block.children, dim=block.dim)
            if isinstance(block, TextBlock) and len(block.rows) == 1 and block.rows[0][0] == "":
                return TextBlock("", [(name, block.rows[0][1])], dim=block.dim)
        return GroupBlock(title=f"{name}: {type(value).__name__}", children=blocks)

    def render_members(
        members: Iterable[tuple[str, Any]],
        active_ids: set[int],
        depth: int,
        *,
        render_as_items: bool = False,
    ) -> list[OutputBlock]:
        children: list[OutputBlock] = []
        for name, item in members:
            if render_as_items:
                children.append(GroupBlock(title=name, children=render_value(item, active_ids, depth)))
            else:
                children.append(render_member(name, item, active_ids, depth))
        return children

    def render_group(
        title: str,
        members: Iterable[tuple[str, Any]],
        active_ids: set[int],
        depth: int,
        value: Any,
        *,
        render_as_items: bool = False,
    ) -> list[OutputBlock]:
        if max_depth is not None and depth >= max_depth:
            return [limit_block("maximum depth reached")]
        if id(value) in active_ids:
            return [TextBlock(type(value).__name__, [("", f"<recursive {type(value).__name__}>")])]
        active_ids.add(id(value))
        try:
            return [
                GroupBlock(
                    title=title,
                    children=render_members(
                        members,
                        active_ids,
                        depth + 1,
                        render_as_items=render_as_items,
                    ),
                )
            ]
        finally:
            active_ids.remove(id(value))

    def render_value(value: Any, active_ids: set[int], depth: int) -> list[OutputBlock]:
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
                active_ids,
                depth,
                value,
            )

        if isinstance(value, Mapping):
            return render_group(
                type(value).__name__,
                ((str(key), item) for key, item in value.items()),
                active_ids,
                depth,
                value,
            )

        if isinstance(value, (list, tuple)):
            item_type = type(value[0]).__name__ if value else "Any"
            return render_group(
                f"{type(value).__name__}[{item_type}]  ×{len(value)}",
                ((f"[{index}]", item) for index, item in enumerate(value)),
                active_ids,
                depth,
                value,
                render_as_items=True,
            )

        return [TextBlock(type(value).__name__, [("", text_for(value))])]

    def format_pydantic_model(value: Any) -> list[OutputBlock]:
        return render_value(value, set(), 0)

    return format_pydantic_model


def _register_pydantic_base_model_formatter() -> None:
    try:
        from pydantic import BaseModel
    except ImportError:
        return

    formatter = pydantic_model_formatter()
    registry = global_formatter_registry()
    registry.register_value_formatter(BaseModel, formatter)

    try:
        from pydantic.v1 import BaseModel as PydanticV1BaseModel
    except ImportError:
        return
    if PydanticV1BaseModel is not BaseModel:
        registry.register_value_formatter(PydanticV1BaseModel, formatter)


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
        registry = global_formatter_registry()
        registry.register_value_formatter(np.ndarray, _format_ndarray)
        registry.register_value_formatter(bytes, _format_bytes)
        registry.register_step_formatter(RegionOpener, _region_step_formatter)
        _register_pydantic_base_model_formatter()
        _BUILTINS_REGISTERED = True
