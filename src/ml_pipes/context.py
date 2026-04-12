from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


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
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        raise NotImplementedError


class Store(ContextOp):
    def __init__(self, name: str, index: int | None = None):
        self.name = name
        self.index = index

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        value = self._extract(current)
        return current, context.store(self.name, value)

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        stored_annotations[self.name] = self._extract_annotation(current_output, expand_output_annotation)
        return (Any,), Any if current_output is None else current_output

    def _extract(self, current: Any) -> Any:
        if self.index is None:
            return current
        if not isinstance(current, tuple):
            raise TypeError(f"Store({self.name!r}) cannot index non-tuple value")
        return current[self.index]

    def _extract_annotation(self, annotation: Any | None, expand_output_annotation: Any) -> Any:
        if annotation is None:
            return Any
        if self.index is None:
            return annotation
        parts = expand_output_annotation(annotation)
        if self.index >= len(parts):
            return Any
        return parts[self.index]


class Recall(ContextOp):
    def __init__(self, name: str):
        self.name = name

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        stored = context.load(self.name)
        if isinstance(current, tuple):
            return current + (stored,), context
        return (current, stored), context

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if self.name not in stored_annotations:
            raise validation_error_type(f"Recall({self.name!r}) references a value that was not stored")

        stored_annotation = stored_annotations[self.name]
        if current_output is None:
            return (Any,), (Any, stored_annotation)

        current_parts = expand_output_annotation(current_output)
        return (Any,), current_parts + (stored_annotation,)


class Pick(ContextOp):
    """Selects one or more elements from a tuple by index, discarding the rest.

    A pure control operator: it changes which value flows forward but does not
    transform any data. Commonly used after Store to discard the ResizeTransform
    and keep only the ImagePayload before inference.
    """

    def __init__(self, *indices: int):
        if not indices:
            raise ValueError("Pick requires at least one index")
        self.indices = indices

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        if not isinstance(current, tuple):
            raise TypeError("Pick can only be applied to tuple outputs")

        selected = tuple(current[index] for index in self.indices)
        if len(selected) == 1:
            return selected[0], context
        return selected, context

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        parts = expand_output_annotation(current_output)
        selected = tuple(parts[index] for index in self.indices if index < len(parts))
        if len(selected) != len(self.indices):
            raise validation_error_type("Pick references tuple indices that are not available")
        if len(selected) == 1:
            return (Any,), selected[0]
        return (Any,), selected
