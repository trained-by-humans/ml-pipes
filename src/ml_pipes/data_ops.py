from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from itertools import dropwhile, islice, takewhile
from types import UnionType
from typing import Any, Generic, TypeVar, Union, get_args, get_origin, get_type_hints

from .context import Selector, SelectorPart, _normalize_selector, _select_annotation
from .region import RegionCloser, RegionOpener
from .tracing import InvocationTrace, StepSpan, _NoOpTrace, merge_traces
from .validation import PipelineValidationError, StaticContractUnavailableError, is_annotation_compatible, resolve_operator_contract


StateT = TypeVar("StateT")
CurrentT = TypeVar("CurrentT")
ValueT = TypeVar("ValueT")
MappedT = TypeVar("MappedT")
ItemT = TypeVar("ItemT")
AnyMapping = Mapping[Any, Any]
Mapper = Callable[[ValueT], MappedT]
Predicate = Callable[[ValueT], bool]
KeySelector = Callable[[ItemT], Hashable]


_MISSING = object()
_NONE_TYPE = type(None)


def _close_iterable(iterator: object) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


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
) -> tuple[Any, Any] | None:
    contract = _resolve_unary_callable_contract(function)
    if contract is None or current_annotation in {None, Any}:
        return contract

    input_type, output_type = contract
    source_annotation = _selector_annotation(
        current_annotation,
        src,
        expand_output_annotation=expand_output_annotation,
        validation_error_type=validation_error_type,
        operator_name=operator_name,
    )
    comparable_source = _without_none(source_annotation)
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


def _optional_annotation(annotation: Any) -> Any:
    if annotation is None:
        return _NONE_TYPE
    if annotation is Any:
        return Any | None
    if _is_union_annotation(annotation) and _NONE_TYPE in get_args(annotation):
        return annotation
    return annotation | None


def _annotation_explicitly_allows_none(annotation: Any) -> bool:
    if annotation in {None, _NONE_TYPE}:
        return True
    if not _is_union_annotation(annotation):
        return False
    return any(option in {None, _NONE_TYPE} for option in get_args(annotation))


def _propagate_explicit_none(input_annotation: Any, output_annotation: Any) -> Any:
    if input_annotation in {None, _NONE_TYPE}:
        return _NONE_TYPE
    if input_annotation is Any:
        return _optional_annotation(output_annotation)
    if _annotation_explicitly_allows_none(input_annotation):
        return _optional_annotation(output_annotation)
    return output_annotation


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


class ForEachItem(RegionOpener):
    """Run the enclosed operators once per source item."""

    closing_type = EndForEachItem

    def run_region(
        self,
        current: Iterable[object],
        label: str,
        execute_region: Any,
        trace: InvocationTrace | _NoOpTrace,
        cfg: Any,
    ) -> list[Any]:
        del cfg
        collecting = isinstance(trace, InvocationTrace)
        source = iter(current)
        results: list[Any] = []
        child_traces: list[InvocationTrace] = []
        t_region = time.perf_counter()

        try:
            for value in source:
                child_trace = InvocationTrace() if collecting else _NoOpTrace()
                result, child_trace = execute_region(value, child_trace)
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


class _ClosingIterable(Generic[ItemT]):
    def __init__(self, iterator: Iterator[ItemT], *, parent: object | None = None):
        self._iterator = iterator
        self._parent = parent
        self._closed = False

    def __iter__(self) -> _ClosingIterable[ItemT]:
        return self

    def __next__(self) -> ItemT:
        return next(self._iterator)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_iterable(self._iterator)
        if self._parent is not None:
            _close_iterable(self._parent)


class RequireMappingValue(Generic[StateT]):
    """Drop non-mapping values and store the mapping at the target selector."""

    def __init__(self, as_: Selector, state_factory: Callable[[], StateT]):
        self.as_ = as_
        self.state_factory = state_factory

    def __call__(self, value: AnyMapping | None) -> StateT | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            return None
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
        if current_output in {None, _NONE_TYPE}:
            return (input_type,), _NONE_TYPE
        if current_output is not Any and _is_mapping_annotation(current_output) and not _annotation_explicitly_allows_none(current_output):
            return (input_type,), base_output
        return (input_type,), _optional_annotation(base_output)


class Map(Generic[ValueT, MappedT]):
    """Apply a function to the current value or a selected field."""

    def __init__(
        self,
        fn: Mapper[ValueT, MappedT],
        *,
        src: Selector | None = None,
        as_: Selector | None = None,
    ):
        if src is None and as_ is not None:
            raise ValueError("Map(as_=...) requires src to be set.")
        self.fn = fn
        self.src = src
        self.as_ = src if as_ is None else as_

    def __call__(self, current: CurrentT | None) -> CurrentT | MappedT | None:
        if current is None:
            return None
        if self.src is None:
            return self.fn(current)
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
        fn_contract = _resolve_callable_contract(
            current_output,
            self.src,
            self.fn,
            callable_label="fn",
            current_label="current value",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
        )

        del stored_annotations, expand_output_annotation, validation_error_type
        if self.src is not None:
            return (current_output,), current_output

        if fn_contract is None:
            return (current_output,), _propagate_explicit_none(current_output, Any)

        fn_input_type, fn_output_type = fn_contract
        if current_output is not None and current_output is not Any:
            return (current_output,), _propagate_explicit_none(current_output, fn_output_type)
        return (fn_input_type,), fn_output_type


class RequireValue:
    """Keep the current value only when the selected value exists and is not None."""

    def __init__(self, src: Selector | None = None):
        self.src = src

    def __call__(self, current: CurrentT | None) -> CurrentT | None:
        if current is None:
            return None
        if self.src is None:
            return current
        value = _read_selector_or_missing(current, self.src)
        if value is _MISSING or value is None:
            return None
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        if self.src is None:
            return (current_output,), _propagate_explicit_none(current_output, current_output)
        return (current_output,), _optional_annotation(current_output)


class Filter(Generic[ValueT]):
    """Keep the current value when a predicate matches the current or selected value."""

    def __init__(
        self,
        predicate: Predicate[ValueT],
        *,
        src: Selector | None = None,
    ):
        self.predicate = predicate
        self.src = src

    def __call__(self, current: CurrentT | None) -> CurrentT | None:
        if current is None:
            return None
        if self.src is None:
            value = current
        else:
            value = _read_selector(current, self.src, type(self).__name__)
        return current if self.predicate(value) else None

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
        )

        return (current_output,), _optional_annotation(current_output)


class DropNone:
    """Remove `None` values from an iterable."""

    def __call__(self, items: Iterable[ItemT | None]) -> list[ItemT]:
        return [item for item in items if item is not None]


class Distinct(Generic[ItemT]):
    """Keep only the first item for each distinct key."""

    def __init__(
        self,
        *,
        src: Selector | None = None,
        key: KeySelector[ItemT] | None = None,
    ):
        if src is not None and key is not None:
            raise ValueError("Distinct accepts either src or key, not both.")
        self.src = src
        self.key = key

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        seen: set[object] = set()
        deduped: list[ItemT] = []
        for item in items:
            current_key = self._key_for(item)
            if current_key in seen:
                continue
            seen.add(current_key)
            deduped.append(item)
        return deduped

    def _key_for(self, item: ItemT) -> object:
        if self.src is not None:
            return _read_selector(item, self.src, type(self).__name__)
        if self.key is not None:
            return self.key(item)
        return item

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        item_type = _iterable_item_annotation(current_output)
        input_type = _sequence_input_annotation(current_output)
        if self.key is not None:
            _resolve_callable_contract(
                item_type,
                None,
                self.key,
                callable_label="key",
                current_label="current item",
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
        if self.count < 0:
            raise ValueError("count must be >= 0.")

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        source = iter(items)
        try:
            return list(islice(source, self.count))
        finally:
            _close_iterable(source)


class Skip:
    """Skip the first `count` items from an iterable."""

    def __init__(self, count: int | str):
        self.count = int(count)
        if self.count < 0:
            raise ValueError("count must be >= 0.")

    def __call__(self, items: Iterable[ItemT]) -> Iterable[ItemT]:
        source = iter(items)
        return _ClosingIterable(islice(source, self.count, None), parent=source)


class TakeWhile(Generic[ItemT]):
    """Keep items while the predicate remains true."""

    def __init__(self, predicate: Predicate[ItemT]):
        self.predicate = predicate

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        source = iter(items)
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
        item_type = _iterable_item_annotation(current_output)
        input_type = _sequence_input_annotation(current_output)
        _resolve_callable_contract(
            item_type,
            None,
            self.predicate,
            callable_label="predicate",
            current_label="current item",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_annotation(item_type)


class SkipWhile(Generic[ItemT]):
    """Skip items while the predicate remains true, then keep the rest."""

    def __init__(self, predicate: Predicate[ItemT]):
        self.predicate = predicate

    def __call__(self, items: Iterable[ItemT]) -> list[ItemT]:
        return list(dropwhile(self.predicate, items))

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
            self.predicate,
            callable_label="predicate",
            current_label="current item",
            expand_output_annotation=expand_output_annotation,
            validation_error_type=validation_error_type,
            operator_name=type(self).__name__,
        )
        del stored_annotations, expand_output_annotation, validation_error_type
        return (input_type,), _list_annotation(item_type)
