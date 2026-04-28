from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import Context
    from .tracing import TracingConfig


class RegionOpener:
    """Base class for region-opening operators (Batch, Scatter, ...).

    Subclasses must set `closing_type` to their matching closer class and
    implement `execute_region` with the region-specific execution logic.

    `_step_into_region` in Pipeline calls `execute_region(pipeline, current,
    context, i, trace, cfg)` where *i* is the index of this opener in
    `pipeline.operators`.  The implementation is responsible for finding the
    matching closer, running the region, and returning
    `(result, context, next_i)` where *next_i* is the index to resume from
    after the closer.
    """

    closing_type: type

    def execute_region(
        self,
        pipeline: Any,
        current: Any,
        context: "Context",
        i: int,
        trace: Any,
        cfg: "TracingConfig | None",
    ) -> tuple[Any, "Context", int]:
        raise NotImplementedError


class RegionCloser:
    """Marker base class for region-closing operators (UnBatch, Gather, ...)."""
