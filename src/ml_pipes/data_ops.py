from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import islice, takewhile
from typing import Any, Generic, TypeVar, cast, get_args, get_origin

from ._typing.annotation import (
    combine_annotation_options,
    is_assignable,
    is_iterable_annotation,
    is_mapping_annotation,
    is_union_annotation,
    iterable_annotation,
    list_annotation,
    remove_none_annotation_options_or_any,
    variadic_tuple_item_annotation,
)
from ._typing.inspection import (
    resolve_callable_annotations,
)
from ._typing.signatures import (
    validate_nullary_callable_signature,
    validate_unary_callable_signature,
)
from .control import SHORT_CIRCUIT
from .operator import Operator
from .region import RegionCloser, RegionExecutor, RegionOpener, RegionTraceLike
from .selector import Selector, SelectorInput
from .tracing import PendingSpan, InvocationTrace, StepSpan, TracingConfig, _NoOpTrace, _extract_shape, capture_value, operator_config


StateT = TypeVar("StateT")
CurrentT = TypeVar("CurrentT")
ValueT = TypeVar("ValueT")
MappedT = TypeVar("MappedT")
ItemT = TypeVar("ItemT")
AnyMapping = Mapping[Any, Any]
Mapper = Callable[[ValueT], MappedT]
NullableMapper = Callable[[ValueT], MappedT | None]
Predicate = Callable[[ValueT], bool]
KeySelector = Callable[[ItemT], Hashable]


def _close_iterable(iterator: object) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


# "Item iterable" means a boundary that yields logical items; values like
# mappings and string/bytes-like objects stay value-shaped even though Python
# can iterate over them.
def _is_value_shaped_iterable(value: Iterable[Any]) -> bool:
    return isinstance(value, (str, bytes, bytearray, Mapping))


def _require_item_iterable(
    value: object,
    *,
    operator_name: str,
) -> Iterable[Any]:
    if not isinstance(value, Iterable) or _is_value_shaped_iterable(cast(Iterable[Any], value)):
        raise TypeError(f"{operator_name} requires an item iterable boundary, got {type(value).__name__}")
    return cast(Iterable[Any], value)


def _resolve_selector(
    operator_name: str,
    *,
    name: str,
    value: SelectorInput | None = None,
    required: bool = False,
) -> Selector:
    selector = Selector.from_input(value)
    if required and not selector:
        raise ValueError(f"{operator_name} requires a non-empty {name} selector")
    return selector


def _resolve_iterable_item_annotation(annotation: Any) -> Any:
    annotation = remove_none_annotation_options_or_any(annotation)
    if annotation in {Any, object}:
        return Any
    if is_union_annotation(annotation):
        item_types = tuple(_resolve_iterable_item_annotation(option) for option in get_args(annotation))
        if not item_types or any(item_type is Any for item_type in item_types):
            return Any
        return combine_annotation_options(*item_types)

    if annotation is str:
        return str
    if annotation in {bytes, bytearray}:
        return int

    variadic_item_type = variadic_tuple_item_annotation(annotation)
    if variadic_item_type is not None:
        return variadic_item_type

    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return Any
        return combine_annotation_options(*args)
    if origin is not None:
        args = get_args(annotation)
        try:
            if issubclass(origin, Mapping):
                return args[0] if args else Any
            if issubclass(origin, Iterable):
                return args[0] if args else Any
        except TypeError:
            return Any
    if isinstance(annotation, type):
        try:
            if issubclass(annotation, Mapping):
                return Any
            if issubclass(annotation, Iterable):
                return Any
        except TypeError:
            return Any
    return Any


def _is_value_shaped_iterable_annotation(annotation: Any) -> bool:
    annotation = remove_none_annotation_options_or_any(annotation)
    if annotation in {Any, object}:
        return False
    if is_union_annotation(annotation):
        options = get_args(annotation)
        return any(_is_value_shaped_iterable_annotation(option) for option in options)

    if annotation in {str, bytes, bytearray}:
        return True

    origin = get_origin(annotation)
    if origin is not None:
        try:
            return issubclass(origin, Mapping)
        except TypeError:
            return False
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, (str, bytes, bytearray, Mapping))
        except TypeError:
            return False
    return False


def _require_iterable_boundary_annotation(
    annotation: Any,
    *,
    operator_name: str,
    validation_error_type: type[Exception],
) -> None:
    if annotation in {Any, object}:
        return
    if not is_iterable_annotation(annotation):
        raise validation_error_type(
            f"{operator_name} requires an iterable boundary, got {annotation}"
        )


def _require_item_iterable_boundary_annotation(
    annotation: Any,
    *,
    operator_name: str,
    validation_error_type: type[Exception],
) -> None:
    _require_iterable_boundary_annotation(
        annotation,
        operator_name=operator_name,
        validation_error_type=validation_error_type,
    )
    if _is_value_shaped_iterable_annotation(annotation):
        raise validation_error_type(
            f"{operator_name} requires an item iterable boundary, got {annotation}"
        )


def _require_callable_annotation(
    annotation: Any | None,
    *,
    operator_name: str,
    callable_label: str,
    annotation_label: str,
    validation_error_type: type[Exception],
) -> Any:
    if annotation is not None:
        return annotation
    raise validation_error_type(
        f"{operator_name} {callable_label} must define a usable {annotation_label} annotation"
    )


def _require_assignment_compatible(
    value_annotation: Any | None,
    target_annotation: Any,
    *,
    operator_name: str,
    source_label: str,
    target_label: str,
    validation_error_type: type[Exception],
) -> None:
    if (
        value_annotation is None
        or target_annotation is None
        or value_annotation is Any
        or target_annotation is Any
    ):
        return
    if is_assignable(value_annotation, target_annotation):
        return
    raise validation_error_type(
        f"{operator_name} {target_label} expects {target_annotation} "
        f"but {source_label} resolves to {value_annotation}"
    )


@Operator
class CollectItems(RegionCloser[ItemT, list[ItemT]]):
    """Region boundary that materializes per-item outputs as a list."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        output_type = list_annotation(current_output)
        return (Any,), output_type


def _mark_trace_dropped(trace: InvocationTrace) -> None:
    if not trace.spans:
        return
    span = trace.spans[-1]
    span.attributes = dict(span.attributes)
    span.attributes["dropped"] = int(span.attributes.get("dropped", 0)) + 1
    if span.child_trace is not None:
        _mark_trace_dropped(span.child_trace)


def _read_dropped(attributes: dict[str, Any]) -> int:
    value = attributes.get("dropped", 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _merge_item_span_attributes(group: list[StepSpan]) -> dict[str, Any]:
    merged = dict(group[0].attributes)
    merged["dropped"] = sum(_read_dropped(span.attributes) for span in group)
    return merged


def _merge_item_traces(traces: list[InvocationTrace]) -> InvocationTrace:
    if not traces:
        return InvocationTrace()
    n = len(traces)
    traces_with_batch_size = [trace for trace in traces if trace.batch_size is not None]
    grouped_spans: dict[str, list[StepSpan]] = {}
    for trace in traces:
        for span in trace.spans:
            grouped_spans.setdefault(span.label, []).append(span)
    spans = [
        StepSpan(
            label=label,
            start_time=0.0,
            duration_s=sum(span.duration_s for span in group) / n,
            error=any(span.error for span in group),
            operator_config=group[0].operator_config,
            attributes=_merge_item_span_attributes(group),
            input_shape=group[0].input_shape,
            output_shape=group[0].output_shape,
            output_value=group[0].output_value,
            child_trace=(
                _merge_item_traces([span.child_trace for span in group if span.child_trace is not None])
                if any(span.child_trace is not None for span in group)
                else None
            ),
            operator_type=group[0].operator_type,
        )
        for label, group in grouped_spans.items()
    ]
    return InvocationTrace(
        spans=spans,
        total_duration_s=sum(trace.total_duration_s for trace in traces) / n,
        batch_size=(
            sum(trace.batch_size for trace in traces_with_batch_size) / len(traces_with_batch_size)
            if traces_with_batch_size
            else None
        ),
        workers=traces[0].workers,
    )


class _MeasuredIterable:
    def __init__(
        self,
        current: Iterable[object],
        execute_region: RegionExecutor[Any, Any],
        span: PendingSpan | None,
        cfg: TracingConfig | None,
    ) -> None:
        self._source = iter(current)
        self._execute_region = execute_region
        self._span = span
        self._cfg = cfg
        self._item_traces: list[InvocationTrace] | None = [] if span is not None else None
        self._item_trace_duration_s = 0.0
        self._seen = 0
        self._emitted = 0
        self._dropped = 0
        self._closed_early = False
        self._source_closed = False
        self._finalized = False

    def __iter__(self) -> "_MeasuredIterable":
        return self

    def __next__(self) -> Any:
        if self._finalized:
            raise StopIteration

        while True:
            try:
                value = next(self._source)
            except StopIteration:
                self._finalize()
                raise

            self._seen += 1
            collecting = self._is_collecting()
            child_trace = InvocationTrace() if collecting else _NoOpTrace()
            try:
                result, child_trace = self._execute_region(value, child_trace)
            except Exception:
                if collecting:
                    incoming = cast(InvocationTrace, child_trace)
                    if self._item_traces is not None:
                        self._item_traces.append(incoming)
                        self._item_trace_duration_s += incoming.total_duration_s
                self._finalize(error=True)
                raise

            if collecting:
                incoming = cast(InvocationTrace, child_trace)
                if result is SHORT_CIRCUIT:
                    _mark_trace_dropped(incoming)
                if self._item_traces is not None:
                    self._item_traces.append(incoming)
                    self._item_trace_duration_s += incoming.total_duration_s

            if result is SHORT_CIRCUIT:
                self._dropped += 1
                continue

            self._emitted += 1
            return result

    def close(self) -> None:
        if self._finalized:
            return
        self._closed_early = True
        self._finalize()

    def _disable_tracing(self) -> None:
        self._span = None
        self._cfg = None
        self._item_traces = None

    def _is_collecting(self) -> bool:
        if self._span is None:
            return False
        if self._span.is_closed:
            self._disable_tracing()
            return False
        if self._item_traces is None:
            self._item_traces = []
        return True

    def _close_source(self) -> Exception | None:
        if self._source_closed:
            return None
        self._source_closed = True
        try:
            _close_iterable(self._source)
        except Exception as exc:
            return exc
        return None

    def _finalize(self, *, error: bool = False) -> None:
        if self._finalized:
            return
        self._finalized = True
        # Finalization must remain best-effort even when source cleanup fails so
        # traces/callbacks are not lost during early termination.
        close_error = self._close_source()
        span = self._span
        cfg = self._cfg
        item_traces = self._item_traces

        self._source = iter(())
        self._execute_region = None
        self._disable_tracing()

        if span is not None and not span.is_closed:
            child_trace = _merge_item_traces(item_traces) if item_traces else None
            span.duration_s = self._item_trace_duration_s
            span.error = error or close_error is not None
            span.attributes = {
                "seen": self._seen,
                "emitted": self._emitted,
                "dropped": self._dropped,
                "closed_early": self._closed_early,
            }
            if cfg and cfg.capture_shapes:
                span.output_shape = f"list [{self._emitted}]"
            if child_trace is not None:
                span.child_trace = child_trace


@Operator
class PerItem(RegionOpener[Iterable[ItemT], ItemT]):
    """Run the enclosed operators once per item from the current item iterable boundary.

    Items that short-circuit inside the region are treated as dropped and are
    omitted from the resulting collection.
    """

    closing_type = CollectItems

    def run_region(
        self,
        current: Iterable[ItemT],
        label: str,
        execute_region: RegionExecutor[ItemT, Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> list[Any]:
        collecting = isinstance(trace, InvocationTrace)
        source = iter(cast(Iterable[ItemT], _require_item_iterable(current, operator_name=type(self).__name__)))
        results: list[Any] = []
        child_traces: list[InvocationTrace] = []
        seen = 0
        emitted = 0
        dropped = 0
        t_region = time.perf_counter()
        child_trace: InvocationTrace | _NoOpTrace | None = None

        try:
            for value in source:
                seen += 1
                child_trace = InvocationTrace() if collecting else _NoOpTrace()
                result, child_trace = execute_region(value, child_trace)
                if collecting:
                    incoming = cast(InvocationTrace, child_trace)
                    if result is SHORT_CIRCUIT:
                        _mark_trace_dropped(incoming)
                    child_traces.append(incoming)
                if result is SHORT_CIRCUIT:
                    dropped += 1
                    child_trace = None
                    continue
                emitted += 1
                results.append(result)
                child_trace = None
        except Exception:
            if collecting and child_trace is not None:
                child_traces.append(cast(InvocationTrace, child_trace))
            merged_trace = _merge_item_traces(child_traces) if child_traces else None
            trace.spans.append(
                StepSpan(
                    label,
                    t_region,
                    time.perf_counter() - t_region,
                    error=True,
                    attributes={
                        "seen": seen,
                        "emitted": emitted,
                        "dropped": dropped,
                    },
                    child_trace=merged_trace if collecting else None,
                    operator_type=type(self),
                )
            )
            raise
        finally:
            _close_iterable(source)

        merged_trace = _merge_item_traces(child_traces) if child_traces else None
        trace.spans.append(
            StepSpan(
                label,
                t_region,
                time.perf_counter() - t_region,
                operator_config=operator_config(self) if (cfg and cfg.capture_config) else {},
                attributes={
                    "seen": seen,
                    "emitted": emitted,
                    "dropped": dropped,
                },
                input_shape=_extract_shape(current) if (cfg and cfg.capture_shapes) else None,
                output_shape=_extract_shape(results) if (cfg and cfg.capture_shapes) else None,
                output_value=capture_value(results) if (cfg and cfg._capture_outputs) else None,
                child_trace=merged_trace if collecting else None,
                operator_type=type(self),
            )
        )
        return results

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        _require_item_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        output_type = _resolve_iterable_item_annotation(current_output)
        return (input_type,), output_type


@Operator
class StreamItems(RegionCloser[ItemT, Iterable[ItemT]]):
    """Region boundary that exposes per-item outputs as a lazy item iterable."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        output_type = iterable_annotation(current_output)
        return (Any,), output_type


@Operator
class LazyPerItem(RegionOpener[Iterable[ItemT], ItemT]):
    """Run the enclosed operators once per item from the current item iterable boundary."""

    closing_type = StreamItems

    def run_region(
        self,
        current: Iterable[ItemT],
        label: str,
        execute_region: RegionExecutor[ItemT, Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> _MeasuredIterable:
        current = cast(Iterable[ItemT], _require_item_iterable(current, operator_name=type(self).__name__))
        span = None
        if isinstance(trace, InvocationTrace):
            span = PendingSpan(
                label,
                time.perf_counter(),
                0.0,
                operator_config=operator_config(self) if (cfg and cfg.capture_config) else {},
                input_shape=_extract_shape(current) if (cfg and cfg.capture_shapes) else None,
                operator_type=type(self),
            )
            cast(list[Any], trace.spans).append(span)
        return _MeasuredIterable(current, execute_region, span, cfg)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        _require_item_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        output_type = _resolve_iterable_item_annotation(current_output)
        return (input_type,), output_type


@Operator
class WrapMappingInObject(Generic[StateT]):
    """Create a new object and store the current mapping at the target selector."""

    def __init__(
        self,
        *,
        target: SelectorInput,
        state_factory: Callable[[], StateT],
    ):
        target_selector = _resolve_selector(
            type(self).__name__,
            name="target",
            value=target,
            required=True,
        )
        validate_nullary_callable_signature(
            state_factory,
            label=f"{type(self).__name__} state_factory",
            error_type=TypeError,
        )
        self._target = target_selector
        self.state_factory = state_factory

    def __call__(self, value: AnyMapping | None) -> StateT:
        if value is None:
            return SHORT_CIRCUIT
        if not isinstance(value, Mapping):
            return SHORT_CIRCUIT
        state = self.state_factory()
        target = self._target.select_field(state)
        target.set(
            value,
            create_missing_mappings=True,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        return state

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        factory_annotations = resolve_callable_annotations(self.state_factory)
        input_type = current_output if is_mapping_annotation(current_output) else AnyMapping | None
        base_output = _require_callable_annotation(
            factory_annotations.return_annotation,
            operator_name=type(self).__name__,
            callable_label="state_factory",
            annotation_label="return type",
            validation_error_type=validation_error_type,
        )
        target_annotation = self._target.validate_write(
            base_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        if is_mapping_annotation(current_output):
            _require_assignment_compatible(
                current_output,
                target_annotation,
                operator_name=type(self).__name__,
                source_label="current mapping",
                target_label=f"target {self._target!r}",
                validation_error_type=validation_error_type,
            )
        return (input_type,), base_output


@Operator
class Map(Generic[ValueT, MappedT]):
    """Apply a function to the current value."""

    def __init__(self, fn: Mapper[ValueT, MappedT]):
        validate_unary_callable_signature(
            fn,
            label=f"{type(self).__name__} fn",
            argument_label="the current value",
            error_type=TypeError,
        )
        self.fn = fn

    def __call__(self, current: ValueT) -> MappedT:
        return self.fn(current)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        fn_annotations = resolve_callable_annotations(self.fn)
        if current_output is Any:
            input_type = _require_callable_annotation(
                fn_annotations.parameter_annotations[0],
                operator_name=type(self).__name__,
                callable_label="fn",
                annotation_label="input type",
                validation_error_type=validation_error_type,
            )
        else:
            input_type = current_output
            _require_assignment_compatible(
                current_output,
                fn_annotations.parameter_annotations[0],
                operator_name=type(self).__name__,
                source_label="current value",
                target_label="fn",
                validation_error_type=validation_error_type,
            )
        output_type = _require_callable_annotation(
            fn_annotations.return_annotation,
            operator_name=type(self).__name__,
            callable_label="fn",
            annotation_label="return type",
            validation_error_type=validation_error_type,
        )

        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), output_type


@Operator
class MapNotNull(Generic[ValueT, MappedT]):
    """Apply a function and short-circuit when the mapped result is None."""

    def __init__(self, fn: NullableMapper[ValueT, MappedT]):
        map_operator = Map(fn)
        self.fn = fn
        self.map = map_operator

    def __call__(self, current: ValueT) -> MappedT:
        mapped = self.map(current)
        if mapped is None:
            return SHORT_CIRCUIT
        return mapped

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        input_types, mapped_output = self.map.resolve_contract(
            current_output,
            stored_annotations,
            expand_output_annotation,
            validation_error_type,
        )
        return input_types, remove_none_annotation_options_or_any(mapped_output)


@Operator
class MapValue(Generic[ValueT, MappedT]):
    """Apply a function to a source value and store the result on the current object."""

    def __init__(
        self,
        fn: Mapper[ValueT, MappedT],
        *,
        source: SelectorInput,
        target: SelectorInput | None = None,
    ):
        source_selector = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )
        if target is None:
            target_selector = source_selector
        else:
            target_selector = _resolve_selector(
                type(self).__name__,
                name="target",
                value=target,
                required=True,
            )
        validate_unary_callable_signature(
            fn,
            label=f"{type(self).__name__} fn",
            argument_label=f"source {source_selector!r}",
            error_type=TypeError,
        )
        self.fn = fn
        self._source = source_selector
        self._target = target_selector

    def __call__(self, current: CurrentT | None) -> CurrentT | None:
        if current is None:
            return None
        value = self._source.select_value(
            current,
            error_prefix=f"{type(self).__name__}(source={self._source!r})",
        )
        target = self._target.select_field(current)
        target.set(
            self.fn(value),
            create_missing_mappings=True,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        source_annotation = self._source.validate_read(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(source={self._source!r})",
        )
        target_annotation = self._target.validate_write(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        fn_annotations = resolve_callable_annotations(self.fn)
        _require_assignment_compatible(
            source_annotation,
            fn_annotations.parameter_annotations[0],
            operator_name=type(self).__name__,
            source_label=f"source {self._source!r}",
            target_label="fn",
            validation_error_type=validation_error_type,
        )
        mapped_annotation = fn_annotations.return_annotation
        if mapped_annotation is not None:
            _require_assignment_compatible(
                mapped_annotation,
                target_annotation,
                operator_name=type(self).__name__,
                source_label="fn return value",
                target_label=f"target {self._target!r}",
                validation_error_type=validation_error_type,
            )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (current_output,), current_output


@Operator
class Filter(Generic[ValueT]):
    """Keep the current value when a predicate matches the current or source value.

    This operator does not treat ``None`` specially. If the current or source
    value can be ``None``, the predicate must decide how to handle it. Use
    ``FilterNotNull`` when null-dropping should be explicit.
    """

    def __init__(
        self,
        predicate: Predicate[ValueT],
        *,
        source: SelectorInput | None = None,
    ):
        source_selector = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
        )
        argument_label = (
            f"source {source_selector!r}"
            if source_selector
            else "the current value"
        )
        validate_unary_callable_signature(
            predicate,
            label=f"{type(self).__name__} predicate",
            argument_label=argument_label,
            error_type=TypeError,
        )
        self.predicate = predicate
        self._source = source_selector

    def __call__(self, current: CurrentT) -> CurrentT:
        if self._source:
            value = self._source.select_value(
                current,
                error_prefix=f"{type(self).__name__}(source={self._source!r})",
            )
        else:
            value = current
        return current if self.predicate(value) else SHORT_CIRCUIT

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations
        source_annotation = current_output
        source_label = "current value"
        if self._source:
            source_annotation = self._source.validate_read(
                current_output,
                validation_error_type=validation_error_type,
                error_prefix=f"{type(self).__name__}(source={self._source!r})",
            )
            source_label = f"source {self._source!r}"
        predicate_annotations = resolve_callable_annotations(self.predicate)
        _require_assignment_compatible(
            source_annotation,
            predicate_annotations.parameter_annotations[0],
            operator_name=type(self).__name__,
            source_label=source_label,
            target_label="predicate",
            validation_error_type=validation_error_type,
        )
        _require_assignment_compatible(
            predicate_annotations.return_annotation,
            bool,
            operator_name=type(self).__name__,
            source_label="predicate return annotation",
            target_label="predicate return type",
            validation_error_type=validation_error_type,
        )

        return (current_output,), current_output


@Operator
class FilterNotNull:
    """Keep the current value only when the source value exists and is not None."""

    def __init__(self, *, source: SelectorInput):
        source_selector = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )
        self._source = source_selector

    def __call__(self, current: CurrentT) -> CurrentT:
        value = self._source.select_value_or_missing(current, missing=None)
        if value is None:
            return SHORT_CIRCUIT
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        self._source.validate_read(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(source={self._source!r})",
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (current_output,), remove_none_annotation_options_or_any(current_output)


@Operator
class DropNull:
    """Short-circuit when the current value is `None`."""

    def __call__(self, current: ItemT | None) -> ItemT:
        if current is None:
            return SHORT_CIRCUIT
        return current


@Operator
class DistinctBy(Generic[ItemT]):
    """Keep only the first item for each computed key."""

    def __init__(self, fn: KeySelector[ItemT]):
        validate_unary_callable_signature(
            fn,
            label=f"{type(self).__name__} fn",
            argument_label="the current item",
            error_type=TypeError,
        )
        self.fn = fn

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        source = iter(items)
        seen: set[Hashable] = set()
        deduped: list[ItemT] = []
        try:
            for item in source:
                current_key = self.fn(item)
                if current_key in seen:
                    continue
                seen.add(current_key)
                deduped.append(item)
            return deduped
        finally:
            _close_iterable(source)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        _require_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        item_type = _resolve_iterable_item_annotation(current_output)
        fn_annotations = resolve_callable_annotations(self.fn)
        _require_assignment_compatible(
            item_type,
            fn_annotations.parameter_annotations[0],
            operator_name=type(self).__name__,
            source_label="current item",
            target_label="fn",
            validation_error_type=validation_error_type,
        )
        _require_assignment_compatible(
            fn_annotations.return_annotation,
            Hashable,
            operator_name=type(self).__name__,
            source_label="fn return annotation",
            target_label="fn return type",
            validation_error_type=validation_error_type,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), list_annotation(item_type)


@Operator
class Distinct(Generic[ItemT]):
    """Keep only the first item for each distinct source value."""

    def __init__(self, *, source: SelectorInput):
        source_selector = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )
        self._source = source_selector

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        source = iter(items)
        seen: set[Hashable] = set()
        deduped: list[ItemT] = []
        try:
            for item in source:
                key = cast(
                    Hashable,
                    self._source.select_value(
                        item,
                        error_prefix=f"{type(self).__name__}(source={self._source!r})",
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            return deduped
        finally:
            _close_iterable(source)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        _require_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        item_type = _resolve_iterable_item_annotation(current_output)
        source_annotation = self._source.validate_read(
            item_type,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(source={self._source!r})",
        )
        _require_assignment_compatible(
            source_annotation,
            Hashable,
            operator_name=type(self).__name__,
            source_label=f"source {self._source!r} annotation",
            target_label="distinct key type",
            validation_error_type=validation_error_type,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), list_annotation(item_type)


@Operator
class Take:
    """Materialize the first `count` items from an iterable boundary."""

    def __init__(self, count: int | str):
        resolved_count = int(count)
        if resolved_count < 0:
            raise ValueError("count must be >= 0.")
        self.count = resolved_count

    def __call__(self, current: Iterable[ItemT]) -> list[ItemT]:
        source = iter(current)
        try:
            return list(islice(source, self.count))
        finally:
            _close_iterable(source)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        _require_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        item_type = _resolve_iterable_item_annotation(current_output)
        output_type = list_annotation(item_type)
        return (input_type,), output_type


@Operator
class TakeWhile(Generic[ItemT]):
    """Materialize items from an iterable boundary while the predicate remains true."""

    def __init__(self, predicate: Predicate[ItemT]):
        validate_unary_callable_signature(
            predicate,
            label=f"{type(self).__name__} predicate",
            argument_label="the current item",
            error_type=TypeError,
        )
        self.predicate = predicate

    def __call__(self, current: Iterable[ItemT]) -> list[ItemT]:
        source = iter(current)
        try:
            return list(takewhile(self.predicate, source))
        finally:
            _close_iterable(source)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        _require_iterable_boundary_annotation(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        input_type = current_output
        item_type = _resolve_iterable_item_annotation(current_output)
        predicate_annotations = resolve_callable_annotations(self.predicate)
        _require_assignment_compatible(
            item_type,
            predicate_annotations.parameter_annotations[0],
            operator_name=type(self).__name__,
            source_label="current item",
            target_label="predicate",
            validation_error_type=validation_error_type,
        )
        _require_assignment_compatible(
            predicate_annotations.return_annotation,
            bool,
            operator_name=type(self).__name__,
            source_label="predicate return annotation",
            target_label="predicate return type",
            validation_error_type=validation_error_type,
        )
        output_type = list_annotation(item_type)
        return (input_type,), output_type
