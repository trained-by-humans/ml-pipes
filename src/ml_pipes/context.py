from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, get_type_hints

SelectorPart = str | int
Selector = SelectorPart | tuple[SelectorPart, ...]


def _normalize_selector(selector: Selector | None) -> tuple[SelectorPart, ...]:
    if selector is None:
        return ()
    if isinstance(selector, tuple):
        return selector
    if isinstance(selector, int):
        return (selector,)
    parts: list[SelectorPart] = []
    for part in selector.split("."):
        if part == "":
            raise ValueError("Store selector cannot contain empty path segments")
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def _selector_label(index: int | None, select: Selector | None, selector: tuple[SelectorPart, ...]) -> str:
    if index is not None:
        return f"index={index}"
    if select is not None:
        return f"select={selector!r}"
    return ""


def _select_annotation(
    annotation: Any | None,
    selector: tuple[SelectorPart, ...],
    expand_output_annotation: Any,
    validation_error_type: type[Exception] | None,
    store_name: str,
    selector_label: str,
) -> Any:
    current = annotation
    for part in selector:
        if current is None or current is Any:
            return Any
        if isinstance(part, int):
            parts = expand_output_annotation(current)
            if part >= len(parts):
                if validation_error_type is not None:
                    raise validation_error_type(
                        f"Store({store_name!r}, {selector_label}) is out of bounds "
                        f"for {current} (length {len(parts)})"
                    )
                return Any
            current = parts[part]
            continue
        current = _attribute_annotation(current, part)
    return current


def _attribute_annotation(annotation: Any, attribute: str) -> Any:
    owner = annotation if isinstance(annotation, type) else None
    if owner is None:
        return Any

    property_obj = getattr(owner, attribute, None)
    if isinstance(property_obj, property):
        try:
            return get_type_hints(property_obj.fget).get("return", Any)
        except Exception:
            return Any

    try:
        return get_type_hints(owner).get(attribute, Any)
    except Exception:
        return Any


@dataclass(frozen=True)
class Context:
    values: Mapping[str, Any] = field(default_factory=dict)

    def store(self, name: str, value: Any) -> "Context":
        merged = dict(self.values)
        merged[name] = value
        return Context(merged)

    def with_values(self, **values: Any) -> "Context":
        merged = dict(self.values)
        merged.update(values)
        return Context(merged)

    def load(self, name: str) -> Any:
        if name not in self.values:
            raise KeyError(f"Context value not found: {name}")
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def merge(self, other: Mapping[str, Any]) -> "Context":
        merged = dict(self.values)
        merged.update(other)
        return Context(merged)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def with_metadata(self, **metadata: Any) -> "Context":
        merged = dict(self.values)
        merged.update(metadata)
        return Context(merged)


class ContextOp(ABC):
    @abstractmethod
    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        raise NotImplementedError

    @abstractmethod
    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        raise NotImplementedError


class Store(ContextOp):
    def __init__(
        self,
        name: str,
        index: int | None = None,
        select: Selector | None = None,
    ):
        if index is not None and select is not None:
            raise ValueError("Store accepts either index or select, not both")
        self.name = name
        self.index = index
        self.select = select
        self.selector = _normalize_selector(select if select is not None else index)
        self.selector_label = _selector_label(index, select, self.selector)

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        value = self._extract(current)
        return current, context.store(self.name, value)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        stored_annotations[self.name] = self._extract_annotation(
            current_output,
            expand_output_annotation,
            validation_error_type,
        )
        return (Any,), current_output

    def _extract(self, current: Any) -> Any:
        if not self.selector:
            return current
        selected = current
        for part in self.selector:
            if isinstance(part, int):
                if not isinstance(selected, tuple):
                    raise TypeError(f"Store({self.name!r}, {self.selector_label}) cannot index non-tuple value")
                selected = selected[part]
                continue
            selected = getattr(selected, part)
        return selected

    def _extract_annotation(
        self,
        annotation: Any | None,
        expand_output_annotation: Any,
        validation_error_type: type[Exception] | None = None,
    ) -> Any:
        if not self.selector:
            return annotation
        return _select_annotation(
            annotation,
            self.selector,
            expand_output_annotation,
            validation_error_type,
            self.name,
            self.selector_label,
        )


class Recall(ContextOp):
    def __init__(self, name: str, index: int | None = None):
        self.name = name
        self.index = index

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        stored = context.load(self.name)
        current_tuple = current if isinstance(current, tuple) else (current,)
        if self.index is None:
            result = current_tuple + (stored,)
        else:
            result = current_tuple[:self.index] + (stored,) + current_tuple[self.index:]
        return result, context

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        stored_annotation = stored_annotations.get(self.name, Any)
        current_parts = expand_output_annotation(current_output)
        if self.index is None:
            result_parts = current_parts + (stored_annotation,)
        else:
            result_parts = current_parts[:self.index] + (stored_annotation,) + current_parts[self.index:]
        return (Any,), result_parts
