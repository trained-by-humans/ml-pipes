from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from ml_pipes.inspection._global_registry import (
    register_step_formatter,
    register_value_formatter,
)
from ml_pipes.region import RegionOpener
from ml_pipes.tracing import StepSpan, _fmt_batch_size
from ml_pipes.inspection.views import (
    ImageBlock,
    OutputBlock,
    StepView,
    TextBlock,
    _build_span_metadata,
)


def _is_rgb_image_array(value: np.ndarray) -> bool:
    return value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3


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


_BUILTINS_LOCK = Lock()
_BUILTINS_REGISTERED = False


def _format_ndarray(value: np.ndarray) -> list[OutputBlock]:
    if _is_rgb_image_array(value):
        height, width = value.shape[:2]
        return [
            ImageBlock(title=f"ndarray  {width}×{height}  RGB", array=value),
            TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))]),
        ]
    return [TextBlock("ndarray", [("shape", str(value.shape)), ("dtype", str(value.dtype))])]


def _format_bytes(value: bytes) -> list[OutputBlock]:
    return [TextBlock("bytes", [("size", f"{len(value) / 1024:.1f} KB")])]


def _region_step_formatter(
    span: StepSpan,
    last_image: np.ndarray | None,
) -> tuple[StepView, np.ndarray | None]:
    return StepView(span.label, _build_span_metadata(span), _region_summary_block(span)), last_image


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
        _BUILTINS_REGISTERED = True
