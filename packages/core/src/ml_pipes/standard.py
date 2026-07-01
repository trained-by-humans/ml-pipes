from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar, cast, get_args, get_origin, overload

from .batch import BatchGate, LeaderBatch
from .context import Recall, Store
from .control import SHORT_CIRCUIT
from .data_ops import (
    CollectItems,
    Distinct,
    DistinctBy,
    DropNull,
    Filter,
    FilterNotNull,
    LazyPerItem,
    Map,
    MapNotNull,
    MapValue,
    PerItem,
    StreamItems,
    Take,
    TakeWhile,
    WrapMappingInObject,
)
from .operator import Operator
from .region import RegionCloser, RegionExecutor, RegionOpener, RegionTraceLike
from .scatter import ScatterGate
from .selector import Selector, SelectorInput
from .tracing import InvocationTrace, StepSpan, TracingConfig, _NoOpTrace, merge_traces
from .validation import PipelineValidationError

__all__ = [
    "Batch",
    "CollectItems",
    "Distinct",
    "DistinctBy",
    "DropNull",
    "Filter",
    "FilterNotNull",
    "Gather",
    "LazyPerItem",
    "Map",
    "MapNotNull",
    "MapValue",
    "PerItem",
    "Pick",
    "Recall",
    "Scatter",
    "Select",
    "SideEffectOp",
    "Store",
    "StreamItems",
    "Take",
    "TakeWhile",
    "UnBatch",
    "WrapMappingInObject",
]

BatchItemT = TypeVar("BatchItemT")
ScatterItemT = TypeVar("ScatterItemT")
PayloadT = TypeVar("PayloadT")
PickIndexT = TypeVar("PickIndexT", bound=int)
PickFirstT = TypeVar("PickFirstT")
PickSecondT = TypeVar("PickSecondT")


@Operator
class Select:
    def __init__(self, *path: SelectorInput):
        self._selector = Selector.from_input(path)
        if not self._selector:
            raise ValueError("Select requires at least one selector part")

    def __call__(self, current: Any) -> Any:
        return self._selector.select_value(
            current,
            error_prefix=f"Select({self._selector!r})",
        )

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations
        if current_output is Any:
            return (Any,), Any

        selected = self._selector.validate_read(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"Select({self._selector!r})",
        )
        return (current_output,), selected


@Operator
class Pick(Generic[PickIndexT]):
    @overload
    def __init__(self: "Pick[Literal[0]]", index: Literal[0]) -> None:
        ...

    @overload
    def __init__(self: "Pick[Literal[1]]", index: Literal[1]) -> None:
        ...

    @overload
    def __init__(self: "Pick[int]", *indices: int) -> None:
        ...

    def __init__(self, *indices: int):
        if not indices:
            raise ValueError("Pick requires at least one index")
        self.indices = indices

    @overload
    def __call__(
        self: "Pick[Literal[0]]",
        current: tuple[PickFirstT, PickSecondT],
    ) -> PickFirstT:
        ...

    @overload
    def __call__(
        self: "Pick[Literal[1]]",
        current: tuple[PickFirstT, PickSecondT],
    ) -> PickSecondT:
        ...

    @overload
    def __call__(self, current: tuple[Any, ...]) -> Any:
        ...

    def __call__(self, current: tuple[Any, ...]) -> Any:
        if not isinstance(current, tuple):
            raise TypeError("Pick can only be applied to tuple outputs")
        selected = tuple(current[index] for index in self.indices)
        if len(selected) == 1:
            return selected[0]
        return selected

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        error_type = validation_error_type or PipelineValidationError

        if current_output is Any or current_output is tuple:
            return (tuple[Any, ...],), Any

        repeated_item = self._homogeneous_tuple_item(current_output)
        if repeated_item is not None:
            selected = tuple(repeated_item for _ in self.indices)
            return (current_output,), selected[0] if len(selected) == 1 else tuple[selected]

        parts = self._fixed_tuple_parts(current_output)
        if parts is None:
            raise error_type(f"Pick requires a tuple boundary, got {current_output}")

        selected = tuple(
            parts[self._normalize_fixed_index(index, len(parts), current_output, error_type)]
            for index in self.indices
        )
        input_annotation = current_output if get_origin(current_output) is tuple else tuple[parts]
        return (input_annotation,), selected[0] if len(selected) == 1 else tuple[selected]

    @staticmethod
    def _fixed_tuple_parts(annotation: Any) -> tuple[Any, ...] | None:
        if isinstance(annotation, tuple):
            return annotation
        if get_origin(annotation) is not tuple:
            return None
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return None
        return args

    @staticmethod
    def _homogeneous_tuple_item(annotation: Any) -> Any | None:
        if get_origin(annotation) is not tuple:
            return None
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return None

    @staticmethod
    def _normalize_fixed_index(
        index: int,
        size: int,
        current_output: Any,
        error_type: type[Exception],
    ) -> int:
        normalized_index = index if index >= 0 else size + index
        if normalized_index < 0 or normalized_index >= size:
            raise error_type(
                f"Pick({index}) is out of bounds for {current_output} (length {size})"
            )
        return normalized_index


class UnBatch(RegionCloser[list[BatchItemT], BatchItemT]):
    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (Any,), args[0] if args else Any
        return (Any,), Any


@Operator
class Batch(RegionOpener[BatchItemT, list[BatchItemT]]):
    closing_type = UnBatch

    def __init__(self, size: int, timeout: float = 0.05) -> None:
        self.gate = BatchGate(size, timeout)

    def run_region(
        self,
        current: BatchItemT,
        label: str,
        execute_region: RegionExecutor[list[BatchItemT], Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> Any:
        gate = self.gate

        t_gate_enter = time.perf_counter()
        outcome = gate.enter(current)
        gate_blocked_duration = time.perf_counter() - t_gate_enter

        if not isinstance(outcome, LeaderBatch):
            batch_region_duration = outcome.batch_span.duration_s if outcome.batch_span is not None else 0.0
            lobby_wait_duration = gate_blocked_duration - batch_region_duration
            trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, lobby_wait_duration))
            if outcome.batch_span is not None:
                trace.spans.append(outcome.batch_span)
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.result

        trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, gate_blocked_duration))
        current = outcome.inputs
        batch_size = len(current) if hasattr(current, "__len__") else None
        collecting = isinstance(trace, InvocationTrace)
        child_trace = InvocationTrace(batch_size=batch_size) if collecting else _NoOpTrace(batch_size=batch_size)

        t_region = time.perf_counter()
        try:
            current, child_trace = execute_region(current, child_trace)
        except Exception as exc:
            error_span = StepSpan(
                label,
                t_region,
                child_trace.total_duration_s,
                error=True,
                child_trace=child_trace if collecting else None,
                operator_type=type(self),
            )
            trace.spans.append(error_span)
            gate.distribute_exception(exc, batch_span=error_span if collecting else None)
            raise

        batch_span = StepSpan(
            label,
            t_region,
            child_trace.total_duration_s,
            child_trace=child_trace if collecting else None,
            operator_type=type(self),
        )
        trace.spans.append(batch_span)
        return gate.distribute(current, batch_span=batch_span if collecting else None)

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


class Gather(RegionCloser[ScatterItemT, list[ScatterItemT]]):
    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


@Operator
class Scatter(RegionOpener[list[ScatterItemT], ScatterItemT]):
    closing_type = Gather

    def __init__(self, max_concurrency: int = 1) -> None:
        self.gate = ScatterGate(max_concurrency)

    def run_region(
        self,
        current: list[ScatterItemT],
        label: str,
        execute_region: RegionExecutor[ScatterItemT, Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> Any:
        gate = self.gate
        collecting = isinstance(trace, InvocationTrace)
        items = current
        n_items = len(items)

        def run_region(entry: Any) -> None:
            child_trace = (
                InvocationTrace(batch_size=n_items, workers=gate.max_concurrency)
                if collecting
                else _NoOpTrace()
            )
            try:
                result, child_trace = execute_region(entry.value, child_trace)
                entry.deposit(result, child_trace if collecting else None)
            except BaseException as exc:
                entry.deposit_exception(exc, child_trace if collecting else None)

        gate.scatter(items, run_region)
        t_gather = time.perf_counter()
        entries, first_exc = gate.gather()
        child_traces = [e.child_trace for e in entries if e.child_trace is not None]
        child_trace = merge_traces(child_traces) if child_traces else None

        if first_exc is not None:
            trace.spans.append(
                StepSpan(
                    label,
                    t_gather,
                    time.perf_counter() - t_gather,
                    error=True,
                    child_trace=child_trace if collecting else None,
                    operator_type=type(self),
                )
            )
            raise first_exc

        trace.spans.append(
            StepSpan(
                label,
                t_gather,
                time.perf_counter() - t_gather,
                child_trace=child_trace if collecting else None,
                operator_type=type(self),
            )
        )
        return [e.result for e in entries if e.result is not SHORT_CIRCUIT]

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (list[Any],), args[0] if args else Any
        return (list[Any],), Any


class SideEffectOp(ABC, Generic[PayloadT]):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override __call__; implement effect() instead"
            )

    @abstractmethod
    def effect(self, payload: PayloadT) -> None:
        raise NotImplementedError

    def __call__(self, payload: PayloadT) -> PayloadT:
        self.effect(payload)
        return payload

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        return (Any,), current_output
