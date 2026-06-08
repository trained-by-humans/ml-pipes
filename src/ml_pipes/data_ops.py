from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import islice, takewhile
from types import UnionType
from typing import Any, Generic, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from .control import SHORT_CIRCUIT
from .operator import Operator
from .region import RegionCloser, RegionOpener
from .selector import Selector, SelectorInput
from .tracing import PendingSpan, InvocationTrace, StepSpan, _NoOpTrace, _extract_shape, capture_value, operator_config
from .validation import PipelineValidationError, StaticContractUnavailableError, is_annotation_compatible, resolve_operator_contract


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

_NONE_TYPE = type(None)


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


def _is_union_annotation(annotation: Any) -> bool:
    return get_origin(annotation) in {UnionType, Union}


def _combine_annotations(*annotations: Any) -> Any:
    if not annotations:
        return Any
    combined = annotations[0]
    for annotation in annotations[1:]:
        combined = combined | annotation
    return combined


def _without_none(annotation: Any) -> Any:
    if annotation in {None, _NONE_TYPE}:
        return Any
    if not _is_union_annotation(annotation):
        return annotation

    remaining = tuple(
        option
        for option in get_args(annotation)
        if option not in {None, _NONE_TYPE}
    )
    if not remaining:
        return Any
    return _combine_annotations(*remaining)


def _is_mapping_annotation(annotation: Any) -> bool:
    annotation = _without_none(annotation)
    if annotation in {Any, object}:
        return False
    if _is_union_annotation(annotation):
        options = get_args(annotation)
        return bool(options) and all(_is_mapping_annotation(option) for option in options)

    origin = get_origin(annotation)
    if origin is not None:
        try:
            return issubclass(origin, Mapping)
        except TypeError:
            return False
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Mapping)
        except TypeError:
            return False
    return False


def _iterable_alias(item_type: Any) -> Any:
    return cast(Any, Iterable)[item_type]


def _list_alias(item_type: Any) -> Any:
    return cast(Any, list)[item_type]


def _resolve_iterable_item_annotation(annotation: Any) -> Any:
    annotation = _without_none(annotation)
    if annotation in {Any, object}:
        return Any
    if _is_union_annotation(annotation):
        item_types = tuple(_resolve_iterable_item_annotation(option) for option in get_args(annotation))
        if not item_types or any(item_type is Any for item_type in item_types):
            return Any
        return _combine_annotations(*item_types)

    if annotation is str:
        return str
    if annotation in {bytes, bytearray}:
        return int

    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return Any
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return _combine_annotations(*args)
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


def _is_iterable_annotation(annotation: Any) -> bool:
    annotation = _without_none(annotation)
    if annotation in {Any, object}:
        return False
    if _is_union_annotation(annotation):
        options = get_args(annotation)
        return bool(options) and all(_is_iterable_annotation(option) for option in options)

    if annotation in {str, bytes, bytearray}:
        return True

    origin = get_origin(annotation)
    if origin is not None:
        try:
            return issubclass(origin, Iterable)
        except TypeError:
            return False
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Iterable)
        except TypeError:
            return False
    return False


def _is_value_shaped_iterable_annotation(annotation: Any) -> bool:
    annotation = _without_none(annotation)
    if annotation in {Any, object}:
        return False
    if _is_union_annotation(annotation):
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


def _resolve_iterable_input_annotation(annotation: Any) -> Any:
    if _is_iterable_annotation(annotation):
        return annotation

    item_type = _resolve_iterable_item_annotation(annotation)
    return _iterable_alias(item_type)


def _resolve_iterable_boundary_contract(
    current_output: Any,
    *,
    operator_name: str,
    validation_error_type: type[Exception],
) -> tuple[Any, Any]:
    if current_output in {Any, object}:
        return Any, Any
    if not _is_iterable_annotation(current_output):
        raise validation_error_type(
            f"{operator_name} requires an iterable boundary, got {current_output}"
        )
    item_type = _resolve_iterable_item_annotation(current_output)
    input_type = _resolve_iterable_input_annotation(current_output)
    return input_type, item_type


def _resolve_per_item_boundary_contract(
    current_output: Any,
    *,
    operator_name: str,
    validation_error_type: type[Exception],
) -> tuple[Any, Any]:
    if current_output in {Any, object}:
        return Any, Any
    if not _is_iterable_annotation(current_output) or _is_value_shaped_iterable_annotation(current_output):
        raise validation_error_type(
            f"{operator_name} requires an item iterable boundary, got {current_output}"
        )
    item_type = _resolve_iterable_item_annotation(current_output)
    input_type = _resolve_iterable_input_annotation(current_output)
    return input_type, item_type


def _resolve_callable_contract(
    source_annotation: Any,
    function: Callable[..., Any],
    *,
    callable_label: str,
    source_label: str,
    validation_error_type: type[Exception],
    operator_name: str,
    ignore_explicit_none: bool = True,
) -> tuple[Any, Any] | None:
    try:
        input_types, output_type = resolve_operator_contract(function)
    except (PipelineValidationError, StaticContractUnavailableError):
        contract = None
    else:
        contract = (input_types[0], output_type) if len(input_types) == 1 else None
    if source_annotation in {None, Any}:
        return contract
    if contract is None:
        return None

    input_type, output_type = contract
    comparable_source = _without_none(source_annotation) if ignore_explicit_none else source_annotation
    if comparable_source is not Any and not is_annotation_compatible(comparable_source, (input_type,)):
        raise validation_error_type(
            f"{operator_name} {callable_label} expects {input_type} "
            f"but {source_label} resolves to {source_annotation}"
        )
    return input_type, output_type


def _resolve_nullary_callable_output(function: Callable[..., Any]) -> Any | None:
    if inspect.isclass(function):
        return function

    try:
        target = function if inspect.isfunction(function) or inspect.ismethod(function) else getattr(function, "__call__")
    except AttributeError:
        return None

    try:
        hints = get_type_hints(target)
        signature = inspect.signature(target)
    except (TypeError, ValueError, NameError):
        return None

    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if parameters:
        return None
    return hints.get("return")


def _require_assignment_compatible(
    value_annotation: Any,
    target_annotation: Any,
    *,
    operator_name: str,
    source_label: str,
    target_label: str,
    validation_error_type: type[Exception],
) -> None:
    if value_annotation is Any or target_annotation is Any:
        return
    if is_annotation_compatible(value_annotation, (target_annotation,)):
        return
    raise validation_error_type(
        f"{operator_name} {target_label} expects {target_annotation} "
        f"but {source_label} resolves to {value_annotation}"
    )


@Operator
class CollectItems(RegionCloser):
    """Region boundary that materializes per-item outputs as a list."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        output_type = _list_alias(_NONE_TYPE if current_output is None else current_output)
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
        execute_region: Any,
        span: PendingSpan | None,
        cfg: Any,
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
class PerItem(RegionOpener):
    """Run the enclosed operators once per item from the current item iterable boundary.

    Items that short-circuit inside the region are treated as dropped and are
    omitted from the resulting collection.
    """

    closing_type = CollectItems

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> list[Any]:
        collecting = isinstance(trace, InvocationTrace)
        source = iter(_require_item_iterable(current, operator_name=type(self).__name__))
        results: list[Any] = []
        child_traces: list[InvocationTrace] = []
        seen = 0
        emitted = 0
        dropped = 0
        t_region = time.perf_counter()

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
                    continue
                emitted += 1
                results.append(result)
        except Exception:
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
        input_type, output_type = _resolve_per_item_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        return (input_type,), output_type


@Operator
class StreamItems(RegionCloser):
    """Region boundary that exposes per-item outputs as a lazy item iterable."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        output_type = _iterable_alias(_NONE_TYPE if current_output is None else current_output)
        return (Any,), output_type


@Operator
class LazyPerItem(RegionOpener):
    """Run the enclosed operators once per item from the current item iterable boundary."""

    closing_type = StreamItems

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> _MeasuredIterable:
        current = _require_item_iterable(current, operator_name=type(self).__name__)
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
        input_type, output_type = _resolve_per_item_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
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
        self._target = _resolve_selector(
            type(self).__name__,
            name="target",
            value=target,
            required=True,
        )
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
        factory_output = _resolve_nullary_callable_output(self.state_factory)
        input_type = current_output if _is_mapping_annotation(current_output) else AnyMapping | None
        base_output = factory_output if factory_output is not None else object
        target_annotation = self._target.validate_write(
            base_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        if _is_mapping_annotation(current_output):
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
        fn_contract = _resolve_callable_contract(
            current_output,
            self.fn,
            callable_label="fn",
            source_label="current value",
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )

        del stored_annotations, expand_output_annotation, validation_error_type
        if fn_contract is None:
            return (current_output,), Any

        fn_input_type, fn_output_type = fn_contract
        if current_output is not Any:
            return (current_output,), fn_output_type
        return (fn_input_type,), fn_output_type


@Operator
class MapNotNull(Generic[ValueT, MappedT]):
    """Apply a function and short-circuit when the mapped result is None."""

    def __init__(self, fn: NullableMapper[ValueT, MappedT]):
        self.fn = fn
        self.map = Map(fn)

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
        return input_types, _without_none(mapped_output)


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
        self.fn = fn
        self._source = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )
        if target is None:
            self._target = self._source
        else:
            self._target = _resolve_selector(
                type(self).__name__,
                name="target",
                value=target,
                required=True,
            )

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
        fn_contract = _resolve_callable_contract(
            source_annotation,
            self.fn,
            callable_label="fn",
            source_label=f"source {self._source!r}",
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        target_annotation = self._target.validate_write(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(target={self._target!r})",
        )
        if fn_contract is not None:
            _, mapped_annotation = fn_contract
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
        self.predicate = predicate
        self._source = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
        )

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
        _resolve_callable_contract(
            source_annotation,
            self.predicate,
            callable_label="predicate",
            source_label=source_label,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )

        return (current_output,), current_output


@Operator
class FilterNotNull:
    """Keep the current value only when the source value exists and is not None."""

    def __init__(self, *, source: SelectorInput):
        self._source = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )

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
        return (current_output,), _without_none(current_output)


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
        input_type, item_type = _resolve_iterable_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        _resolve_callable_contract(
            item_type,
            self.fn,
            callable_label="fn",
            source_label="current item",
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_alias(item_type)


@Operator
class Distinct(Generic[ItemT]):
    """Keep only the first item for each distinct source value."""

    def __init__(self, *, source: SelectorInput):
        self._source = _resolve_selector(
            type(self).__name__,
            name="source",
            value=source,
            required=True,
        )

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
        input_type, item_type = _resolve_iterable_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        self._source.validate_read(
            item_type,
            validation_error_type=validation_error_type,
            error_prefix=f"{type(self).__name__}(source={self._source!r})",
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_alias(item_type)


@Operator
class Take:
    """Materialize the first `count` items from an iterable boundary."""

    def __init__(self, count: int | str):
        self.count = int(count)
        if self.count < 0:
            raise ValueError("count must be >= 0.")

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
        input_type, item_type = _resolve_iterable_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        output_type = _list_alias(item_type)
        return (input_type,), output_type


@Operator
class TakeWhile(Generic[ItemT]):
    """Materialize items from an iterable boundary while the predicate remains true."""

    def __init__(self, predicate: Predicate[ItemT]):
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
        input_type, item_type = _resolve_iterable_boundary_contract(
            current_output,
            operator_name=type(self).__name__,
            validation_error_type=validation_error_type,
        )
        output_type = _list_alias(item_type)
        predicate_input = Any if current_output in {Any, object} else _resolve_iterable_item_annotation(current_output)
        _resolve_callable_contract(
            predicate_input,
            self.predicate,
            callable_label="predicate",
            source_label="current item",
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), output_type
