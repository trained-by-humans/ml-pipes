from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping, Sequence
from itertools import dropwhile, islice, takewhile
from types import UnionType
from typing import Any, Generic, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from .context import Selector, SelectorPart, _normalize_selector, _select_annotation
from .control import SHORT_CIRCUIT
from .region import RegionCloser, RegionOpener
from .tracing import PendingSpan, InvocationTrace, StepSpan, _NoOpTrace, _extract_shape, capture_value, merge_traces, operator_config
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


_MISSING = object()
_NONE_TYPE = type(None)


def _close_iterable(iterator: object) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _is_runtime_sequence_value(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping))


def _select_part(current: object, part: SelectorPart) -> object:
    if isinstance(part, int):
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            try:
                return current[part]
            except IndexError:
                return _MISSING
        return _MISSING
    if isinstance(current, Mapping):
        return current.get(part, _MISSING)
    return getattr(current, part, _MISSING)


def _read_selector_or_missing(value: object, selector: Selector) -> object:
    current = value
    for part in _normalize_selector(selector):
        current = _select_part(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _read_selector_or_none(value: object, selector: Selector) -> object | None:
    selected = _read_selector_or_missing(value, selector)
    if selected is _MISSING:
        return None
    return selected


def _read_selector(value: object, selector: Selector, operator_name: str) -> object:
    selected = _read_selector_or_missing(value, selector)
    if selected is _MISSING:
        raise TypeError(
            f"{operator_name} requires selector {_normalize_selector(selector)!r} on {type(value)!r}."
        )
    return selected


def _write_selector(value: object, selector: Selector, item: object, operator_name: str) -> None:
    parts = _normalize_selector(selector)
    if not parts:
        raise ValueError(f"{operator_name} requires a non-empty selector.")

    current = value
    for part in parts[:-1]:
        next_value = _select_part(current, part)
        if next_value is _MISSING:
            if isinstance(part, str) and isinstance(current, MutableMapping):
                current[part] = {}
                next_value = current[part]
            else:
                raise TypeError(
                    f"{operator_name} cannot resolve parent selector {parts!r} on {type(value)!r}."
                )
        current = next_value

    last = parts[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise TypeError(
                f"{operator_name} can only assign index selectors into list parents, got {type(current)!r}."
            )
        try:
            current[last] = item
        except IndexError as exc:
            raise IndexError(
                f"{operator_name} index selector {last} is out of range for parent list."
            ) from exc
        return

    if isinstance(current, MutableMapping):
        current[last] = item
        return

    try:
        setattr(current, last, item)
    except AttributeError as exc:
        raise TypeError(
            f"{operator_name} requires writable selector {parts!r} on {type(value)!r}."
        ) from exc


def _resolve_unary_callable_contract(function: Callable[..., Any]) -> tuple[Any, Any] | None:
    try:
        input_types, output_type = resolve_operator_contract(function)
    except (PipelineValidationError, StaticContractUnavailableError):
        return None
    if len(input_types) != 1:
        return None
    return input_types[0], output_type


def _resolve_callable_contract(
    current_annotation: Any,
    src: Selector | None,
    function: Callable[..., Any],
    *,
    callable_label: str,
    current_label: str,
    expand_output_annotation: Any,
    validation_error_type: type[Exception],
    operator_name: str,
    ignore_explicit_none: bool = True,
) -> tuple[Any, Any] | None:
    contract = _resolve_unary_callable_contract(function)
    if current_annotation in {None, Any}:
        return contract
    source_annotation = _selector_annotation(
        current_annotation,
        src,
        expand_output_annotation=expand_output_annotation,
        validation_error_type=validation_error_type,
        operator_name=operator_name,
    )
    if contract is None:
        return None

    input_type, output_type = contract
    comparable_source = _without_none(source_annotation) if ignore_explicit_none else source_annotation
    if comparable_source is not Any and not is_annotation_compatible(comparable_source, (input_type,)):
        source_label = (
            f"src selector {_normalize_selector(src)!r}"
            if src is not None
            else current_label
        )
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


def _mapping_input_annotation(annotation: Any) -> Any:
    if _is_mapping_annotation(annotation):
        return annotation
    return AnyMapping | None


def _iterable_item_annotation(annotation: Any) -> Any:
    annotation = _without_none(annotation)
    if annotation in {Any, object}:
        return Any
    if _is_union_annotation(annotation):
        item_types = tuple(_iterable_item_annotation(option) for option in get_args(annotation))
        if not item_types or any(item_type is Any for item_type in item_types):
            return Any
        return _combine_annotations(*item_types)

    origin = get_origin(annotation)
    if origin in {list, Sequence, Iterable}:
        args = get_args(annotation)
        return args[0] if args else Any
    if origin is tuple:
        args = get_args(annotation)
        if not args:
            return Any
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return _combine_annotations(*args)
    if isinstance(annotation, type):
        try:
            if issubclass(annotation, Iterable) and not issubclass(annotation, (str, bytes, bytearray, Mapping)):
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

    origin = get_origin(annotation)
    if origin in {list, Sequence, Iterable, tuple}:
        return True
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Iterable) and not issubclass(annotation, (str, bytes, bytearray, Mapping))
        except TypeError:
            return False
    return False


def _sequence_input_annotation(annotation: Any) -> Any:
    if _is_iterable_annotation(annotation):
        return annotation

    item_type = _iterable_item_annotation(annotation)
    if item_type is Any:
        return Iterable[Any]
    return Iterable[item_type]


def _list_annotation(item_type: Any) -> Any:
    if item_type is Any:
        return list[Any]
    return list[item_type]


def _iterable_annotation(item_type: Any) -> Any:
    if item_type is Any:
        return Iterable[Any]
    return Iterable[item_type]


def _normalize_annotation(annotation: Any) -> Any:
    return _NONE_TYPE if annotation is None else annotation


def _selector_annotation(
    annotation: Any,
    selector: Selector | None,
    *,
    expand_output_annotation: Any,
    validation_error_type: type[Exception] | None,
    operator_name: str,
) -> Any:
    normalized = _normalize_selector(selector)
    if not normalized:
        return annotation
    return _select_annotation(
        annotation,
        normalized,
        expand_output_annotation,
        validation_error_type,
        operator_name,
        f"src={normalized!r}",
    )


class EndForEachItem(RegionCloser):
    """Region boundary for the end of per-item processing."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        output_type = _list_annotation(_normalize_annotation(current_output))
        return (Any,), output_type


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
            child_trace = merge_traces(item_traces) if item_traces else None
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


class ForEachItem(RegionOpener):
    """Run the enclosed operators once per source item.

    Items that short-circuit inside the region are treated as dropped and are
    omitted from the resulting collection.
    """

    closing_type = EndForEachItem

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> list[Any]:
        collecting = isinstance(trace, InvocationTrace)
        source = iter(current)
        results: list[Any] = []
        child_traces: list[InvocationTrace] = []
        t_region = time.perf_counter()

        try:
            for value in source:
                child_trace = InvocationTrace() if collecting else _NoOpTrace()
                result, child_trace = execute_region(value, child_trace)
                if result is not SHORT_CIRCUIT:
                    results.append(result)
                if collecting:
                    child_traces.append(child_trace)
        except Exception:
            merged_trace = merge_traces(child_traces) if child_traces else None
            trace.spans.append(
                StepSpan(
                    label,
                    t_region,
                    time.perf_counter() - t_region,
                    error=True,
                    child_trace=merged_trace if collecting else None,
                    operator_type=type(self),
                )
            )
            raise
        finally:
            _close_iterable(source)

        merged_trace = merge_traces(child_traces) if child_traces else None
        trace.spans.append(
            StepSpan(
                label,
                t_region,
                time.perf_counter() - t_region,
                operator_config=operator_config(self) if (cfg and cfg.capture_config) else {},
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
        del stored_annotations, expand_output_annotation, validation_error_type
        return (_sequence_input_annotation(current_output),), _iterable_item_annotation(current_output)


class EndLazyForEachItem(RegionCloser):
    """Region boundary for lazy per-item processing."""

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        output_type = _iterable_annotation(_normalize_annotation(current_output))
        return (Any,), output_type


class LazyForEachItem(RegionOpener):
    """Run the enclosed operators once per source item and stream results lazily."""

    closing_type = EndLazyForEachItem

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> _MeasuredIterable:
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
        del stored_annotations, expand_output_annotation, validation_error_type
        return (_sequence_input_annotation(current_output),), _iterable_item_annotation(current_output)


class WrapMappingInObject(Generic[StateT]):
    """Create a new object and store the current mapping at the target selector."""

    def __init__(self, as_: Selector, state_factory: Callable[[], StateT]):
        self.as_ = as_
        self.state_factory = state_factory

    def __call__(self, value: AnyMapping | None) -> StateT:
        if value is None:
            return SHORT_CIRCUIT
        if not isinstance(value, Mapping):
            return SHORT_CIRCUIT
        state = self.state_factory()
        _write_selector(state, self.as_, value, type(self).__name__)
        return state

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        factory_output = _resolve_nullary_callable_output(self.state_factory)
        input_type = _mapping_input_annotation(current_output)
        base_output = factory_output if factory_output is not None else object
        return (input_type,), base_output


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
            None,
            self.fn,
            callable_label="fn",
            current_label="current value",
            expand_output_annotation=expand_output_annotation,
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


class MapValue(Generic[ValueT, MappedT]):
    """Apply a function to a selected value and store the result on the current object."""

    def __init__(
        self,
        fn: Mapper[ValueT, MappedT],
        *,
        src: Selector,
        as_: Selector | None = None,
    ):
        self.fn = fn
        self.src = src
        self.as_ = src if as_ is None else as_

    def __call__(self, current: CurrentT | None) -> CurrentT | None:
        if current is None:
            return None
        value = _read_selector(current, self.src, type(self).__name__)
        _write_selector(current, self.as_, self.fn(value), type(self).__name__)
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        _resolve_callable_contract(
            current_output,
            self.src,
            self.fn,
            callable_label="fn",
            current_label="current value",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )

        del stored_annotations, expand_output_annotation, validation_error_type
        return (current_output,), current_output


class Filter(Generic[ValueT]):
    """Keep the current value when a predicate matches the current or selected value.

    This operator does not treat ``None`` specially. If the current or selected
    value can be ``None``, the predicate must decide how to handle it. Use
    ``FilterNotNull`` when null-dropping should be explicit.
    """

    def __init__(
        self,
        predicate: Predicate[ValueT],
        *,
        src: Selector | None = None,
    ):
        self.predicate = predicate
        self.src = src

    def __call__(self, current: CurrentT) -> CurrentT:
        value = self._read_value(current)
        return current if self.predicate(value) else SHORT_CIRCUIT

    def _read_value(self, current: CurrentT) -> object:
        if self.src is None:
            return current
        return _read_selector(current, self.src, type(self).__name__)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations
        _resolve_callable_contract(
            current_output,
            self.src,
            self.predicate,
            callable_label="predicate",
            current_label="current value",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )

        return (current_output,), current_output


class FilterNotNull:
    """Keep the current value only when the selected value exists and is not None."""

    def __init__(self, src: Selector):
        self.src = src

    def __call__(self, current: CurrentT) -> CurrentT:
        value = _read_selector_or_none(current, self.src)
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
        _selector_annotation(
            current_output,
            self.src,
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (current_output,), _without_none(current_output)


class DropNull:
    """Short-circuit when the current value is `None`."""

    def __call__(self, current: ItemT | None) -> ItemT:
        if current is None:
            return SHORT_CIRCUIT
        return current


class DistinctBy(Generic[ItemT]):
    """Keep only the first item for each computed key."""

    def __init__(self, fn: KeySelector[ItemT]):
        self.fn = fn

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        seen: set[Hashable] = set()
        deduped: list[ItemT] = []
        for item in items:
            current_key = self.fn(item)
            if current_key in seen:
                continue
            seen.add(current_key)
            deduped.append(item)
        return deduped

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        item_type = _iterable_item_annotation(current_output)
        input_type = _sequence_input_annotation(current_output)
        _resolve_callable_contract(
            item_type,
            None,
            self.fn,
            callable_label="fn",
            current_label="current item",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_annotation(item_type)


class Distinct(Generic[ItemT]):
    """Keep only the first item for each distinct selected value."""

    def __init__(self, *, src: Selector):
        self.src = src
        self._inner = DistinctBy(self._selector_key(src))

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        return self._inner(items)

    def _selector_key(self, src: Selector) -> KeySelector[ItemT]:
        operator_name = type(self).__name__

        def key(item: ItemT) -> Hashable:
            return cast(Hashable, _read_selector(item, src, operator_name))

        return key

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        item_type = _iterable_item_annotation(current_output)
        input_type = _sequence_input_annotation(current_output)
        _selector_annotation(
            item_type,
            self.src,
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_annotation(item_type)


class Take:
    """Keep the first `count` items from an iterable."""

    def __init__(self, count: int | str):
        self.count = int(count)
        self.remaining = self.count
        if self.count < 0:
            raise ValueError("count must be >= 0.")

    def __call__(self, current: ItemT | Iterable[ItemT] | None) -> ItemT | list[ItemT]:
        if current is None:
            return SHORT_CIRCUIT
        if _is_runtime_sequence_value(current):
            source = iter(current)
            try:
                return list(islice(source, self.count))
            finally:
                _close_iterable(source)
        if self.remaining <= 0:
            return SHORT_CIRCUIT
        self.remaining -= 1
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        if current_output in {Any, object}:
            return (Any,), Any
        if _is_iterable_annotation(current_output):
            item_type = _iterable_item_annotation(current_output)
            input_type = _sequence_input_annotation(current_output)
            return (input_type,), _list_annotation(item_type)
        return (current_output,), _without_none(current_output)


class Skip:
    """Skip the first `count` items from an iterable."""

    def __init__(self, count: int | str):
        self.count = int(count)
        self.remaining = self.count
        if self.count < 0:
            raise ValueError("count must be >= 0.")

    def __call__(self, current: ItemT | Iterable[ItemT] | None) -> ItemT | list[ItemT]:
        if current is None:
            return SHORT_CIRCUIT
        if _is_runtime_sequence_value(current):
            return list(islice(current, self.count, None))
        if self.remaining > 0:
            self.remaining -= 1
            return SHORT_CIRCUIT
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        if current_output in {Any, object}:
            return (Any,), Any
        if _is_iterable_annotation(current_output):
            item_type = _iterable_item_annotation(current_output)
            input_type = _sequence_input_annotation(current_output)
            return (input_type,), _list_annotation(item_type)
        return (current_output,), _without_none(current_output)


class TakeWhile(Generic[ItemT]):
    """Keep items while the predicate remains true."""

    def __init__(self, predicate: Predicate[ItemT]):
        self.predicate = predicate
        self.active = True

    def __call__(self, current: ItemT | Iterable[ItemT] | None) -> ItemT | list[ItemT]:
        if current is None:
            return SHORT_CIRCUIT
        if _is_runtime_sequence_value(current):
            source = iter(current)
            try:
                return list(takewhile(self.predicate, source))
            finally:
                _close_iterable(source)
        if not self.active:
            return SHORT_CIRCUIT
        if not self.predicate(current):
            self.active = False
            return SHORT_CIRCUIT
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if current_output in {Any, object}:
            item_type = Any
            input_type = Any
            predicate_input = Any
        elif _is_iterable_annotation(current_output):
            item_type = _iterable_item_annotation(current_output)
            input_type = _sequence_input_annotation(current_output)
            predicate_input = item_type
        else:
            item_type = _without_none(current_output)
            input_type = current_output
            predicate_input = item_type
        _resolve_callable_contract(
            predicate_input,
            None,
            self.predicate,
            callable_label="predicate",
            current_label="current item",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        if current_output in {Any, object}:
            return (Any,), Any
        if _is_iterable_annotation(current_output):
            return (input_type,), _list_annotation(item_type)
        return (input_type,), _without_none(current_output)


class SkipWhile(Generic[ItemT]):
    """Skip items while the predicate remains true, then keep the rest."""

    def __init__(self, predicate: Predicate[ItemT]):
        self.predicate = predicate
        self.skipping = True

    def __call__(self, current: ItemT | Iterable[ItemT] | None) -> ItemT | list[ItemT]:
        if current is None:
            return SHORT_CIRCUIT
        if _is_runtime_sequence_value(current):
            return list(dropwhile(self.predicate, current))
        if self.skipping and self.predicate(current):
            return SHORT_CIRCUIT
        self.skipping = False
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if current_output in {Any, object}:
            item_type = Any
            input_type = Any
            predicate_input = Any
        elif _is_iterable_annotation(current_output):
            item_type = _iterable_item_annotation(current_output)
            input_type = _sequence_input_annotation(current_output)
            predicate_input = item_type
        else:
            item_type = _without_none(current_output)
            input_type = current_output
            predicate_input = item_type
        _resolve_callable_contract(
            predicate_input,
            None,
            self.predicate,
            callable_label="predicate",
            current_label="current item",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
            ignore_explicit_none=False,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        if current_output in {Any, object}:
            return (Any,), Any
        if _is_iterable_annotation(current_output):
            return (input_type,), _list_annotation(item_type)
        return (input_type,), _without_none(current_output)
