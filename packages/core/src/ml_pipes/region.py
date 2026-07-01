from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from .tracing import InvocationTrace, TracingConfig, _NoOpTrace
    RegionTraceLike: TypeAlias = InvocationTrace | _NoOpTrace
else:
    RegionTraceLike = Any


InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
BodyInputT = TypeVar("BodyInputT", contravariant=True)
BodyOutputT = TypeVar("BodyOutputT", covariant=True)


class RegionExecutor(Protocol[BodyInputT, BodyOutputT]):
    def __call__(
        self,
        value: BodyInputT,
        child_trace: RegionTraceLike,
    ) -> tuple[BodyOutputT, RegionTraceLike]:
        ...


class RegionOpener(Generic[InputT, OutputT]):
    """Base class for region-opening operators (Batch, Scatter, ...).

    The generic parameters describe the local boundary the opener presents to
    static type checkers:
    - ``InputT`` is the value type before stepping into the region.
    - ``OutputT`` is the value type seen by the first operator inside
      the region.

    This is intentionally local and linear: the opener can appear in the
    operator list like any other step, while runtime region execution still
    flows through `run_region`.

    Subclasses set `closing_type` to their matching closer class and implement
    `execute_region`.  `_step_into_region` in Pipeline pre-computes `label` and
    `region`, then calls `execute_region` and wraps the result into the
    `(result, context, next_i)` triple it needs.
    """

    closing_type: type["RegionCloser[Any, Any]"]

    def run_region(
        self,
        current: InputT,
        label: str,
        execute_region: RegionExecutor[OutputT, Any],
        trace: RegionTraceLike,
        cfg: "TracingConfig | None",
    ) -> Any:
        raise NotImplementedError


class RegionCloser(Generic[InputT, OutputT]):
    """Marker base class for region-closing operators (UnBatch, Gather, ...).

    The generic parameters describe the local boundary the closer presents to
    static type checkers:
    - ``InputT`` is the value type seen by the last operator inside the
      region.
    - ``OutputT`` is the value type after leaving the region.
    """
