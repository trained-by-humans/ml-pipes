from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Mapping, Protocol, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class Context:
    transforms: tuple[Any, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def add(self, transform: Any) -> "Context":
        return Context(self.transforms + (transform,), dict(self.metadata))

    def with_metadata(self, **metadata: Any) -> "Context":
        merged = dict(self.metadata)
        merged.update(metadata)
        return Context(self.transforms, merged)


@dataclass(frozen=True)
class Value(Generic[T]):
    data: T
    context: Context = field(default_factory=Context)


class Operator(Protocol):
    def __call__(self, value: Any) -> Any:
        ...


class Pipeline:
    def __init__(self, operators: Iterable[Operator]):
        self.operators = list(operators)

    def __call__(self, value: Any) -> Any:
        current = value
        for operator in self.operators:
            current = operator(current)
        return current
