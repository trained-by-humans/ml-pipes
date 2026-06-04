from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from .operator import Operator
from .selector import Selector, SelectorInput


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


@Operator
class Store(ContextOp):
    def __init__(
        self,
        name: str,
        *,
        source: SelectorInput | None = None,
    ):
        self.name = name
        self._selector = Selector.from_input(source)

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        value = self._selector.select_value(
            current,
            error_prefix=f"Store({self.name!r}, {self._selector!r})",
        )
        return current, context.store(self.name, value)

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        stored_annotations[self.name] = self._selector.validate_read(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"Store({self.name!r}, {self._selector!r})",
        )
        return (Any,), current_output


@Operator
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
