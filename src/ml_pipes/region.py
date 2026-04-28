from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tracing import InvocationTrace, TracingConfig, _NoOpTrace


class RegionOpener:
    """Base class for region-opening operators (Batch, Scatter, ...).

    Subclasses set `closing_type` to their matching closer class and implement
    `execute_region`.  `_step_into_region` in Pipeline pre-computes `label` and
    `region`, then calls `execute_region` and wraps the result into the
    `(result, context, next_i)` triple it needs.
    """

    closing_type: type

    def run_region(
        self,
        current: Any,
        label: str,
        execute_region: Callable,
        trace: "InvocationTrace | _NoOpTrace",
        cfg: "TracingConfig | None",
    ) -> Any:
        raise NotImplementedError


class RegionCloser:
    """Marker base class for region-closing operators (UnBatch, Gather, ...)."""
