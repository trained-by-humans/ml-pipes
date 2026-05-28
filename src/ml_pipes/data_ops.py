from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from itertools import dropwhile, islice, takewhile
from typing import Any

from .context import Selector, SelectorPart, _normalize_selector
from .region import RegionCloser, RegionOpener
from .tracing import InvocationTrace, StepSpan, _NoOpTrace, merge_traces


_MISSING = object()


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


def _write_selector(value: object, selector: Selector, item: Any, operator_name: str) -> None:
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


class EndForEachItem(RegionCloser):
    """Region boundary for the end of per-item processing."""

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        output_type = list[current_output] if current_output is not None else list[Any]
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
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        return (Any,), Any


class RequireMappingValue:
    """Drop non-mapping values and store the mapping at the target selector."""

    def __init__(self, as_: Selector, state_factory: Callable[[], object]):
        self.as_ = as_
        self.state_factory = state_factory

    def __call__(self, value: object | None) -> object | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            return None
        state = self.state_factory()
        _write_selector(state, self.as_, value, type(self).__name__)
        return state


class Map:
    """Apply a function to the current value or a selected field."""

    def __init__(
        self,
        fn: Callable[[Any], Any],
        *,
        src: Selector | None = None,
        as_: Selector | None = None,
    ):
        if src is None and as_ is not None:
            raise ValueError("Map(as_=...) requires src to be set.")
        self.fn = fn
        self.src = src
        self.as_ = src if as_ is None else as_

    def __call__(self, current: object | None) -> object | None:
        if current is None:
            return None
        if self.src is None:
            return self.fn(current)
        value = _read_selector(current, self.src, type(self).__name__)
        _write_selector(current, self.as_, self.fn(value), type(self).__name__)
        return current


class RequireValue:
    """Keep the current value only when the selected value exists and is not None."""

    def __init__(self, src: Selector | None = None):
        self.src = src

    def __call__(self, current: object | None) -> object | None:
        if current is None:
            return None
        if self.src is None:
            return current
        value = _read_selector_or_missing(current, self.src)
        if value is _MISSING or value is None:
            return None
        return current


class Filter:
    """Keep the current value when a predicate matches the current or selected value."""

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        *,
        src: Selector | None = None,
    ):
        self.predicate = predicate
        self.src = src

    def __call__(self, current: object | None) -> object | None:
        if current is None:
            return None
        if self.src is None:
            value = current
        else:
            value = _read_selector(current, self.src, type(self).__name__)
        return current if self.predicate(value) else None


class DropNone:
    """Remove `None` values from an iterable."""

    def __call__(self, items: Iterable[Any | None]) -> list[Any]:
        return [item for item in items if item is not None]


class Distinct:
    """Keep only the first item for each distinct key."""

    def __init__(
        self,
        *,
        src: Selector | None = None,
        key: Callable[[Any], Any] | None = None,
    ):
        if src is not None and key is not None:
            raise ValueError("Distinct accepts either src or key, not both.")
        self.src = src
        self.key = key

    def __call__(self, items: list[Any] | Iterable[Any]) -> list[Any]:
        seen: set[Any] = set()
        deduped: list[Any] = []
        for item in items:
            current_key = self._key_for(item)
            if current_key in seen:
                continue
            seen.add(current_key)
            deduped.append(item)
        return deduped

    def _key_for(self, item: Any) -> Any:
        if self.src is not None:
            return _read_selector(item, self.src, type(self).__name__)
        if self.key is not None:
            return self.key(item)
        return item


class Take:
    """Keep the first `count` items from an iterable."""

    def __init__(self, count: int | str):
        self.count = int(count)
        if self.count < 0:
            raise ValueError("count must be >= 0.")

    def __call__(self, items: list[Any] | Iterable[Any]) -> list[Any]:
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

    def __call__(self, items: list[Any] | Iterable[Any]) -> list[Any]:
        source = iter(items)
        try:
            return list(islice(source, self.count, None))
        finally:
            _close_iterable(source)


class TakeWhile:
    """Keep items while the predicate remains true."""

    def __init__(self, predicate: Callable[[Any], bool]):
        self.predicate = predicate

    def __call__(self, items: list[Any] | Iterable[Any]) -> list[Any]:
        source = iter(items)
        try:
            return list(takewhile(self.predicate, source))
        finally:
            _close_iterable(source)


class SkipWhile:
    """Skip items while the predicate remains true, then keep the rest."""

    def __init__(self, predicate: Callable[[Any], bool]):
        self.predicate = predicate

    def __call__(self, items: list[Any] | Iterable[Any]) -> list[Any]:
        source = iter(items)
        try:
            return list(dropwhile(self.predicate, source))
        finally:
            _close_iterable(source)
