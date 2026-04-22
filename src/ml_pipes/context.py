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
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if self.name not in stored_annotations:
            raise validation_error_type(f"Recall({self.name!r}) references a value that was not stored")

        stored_annotation = stored_annotations[self.name]
        current_parts = expand_output_annotation(current_output) if current_output is not None else (Any,)
        if self.index is None:
            result_parts = current_parts + (stored_annotation,)
        else:
            result_parts = current_parts[:self.index] + (stored_annotation,) + current_parts[self.index:]
        return (Any,), result_parts
