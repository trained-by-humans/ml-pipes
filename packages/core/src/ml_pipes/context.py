from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Mapping, TypeVar, overload

from ml_pipes._typing.annotation import (
    build_union_annotation_from_options,
    expand_annotation_parts,
    variadic_tuple_item_annotation,
)
from ml_pipes.operator import Operator
from ml_pipes.selector import Selector, SelectorInput


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


CurrentT = TypeVar("CurrentT")
InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
StoredT = TypeVar("StoredT")
PrependT = TypeVar("PrependT", bound=bool)


class ContextOp(ABC, Generic[InputT, OutputT]):
    """Operator with a pipeline-visible boundary and hidden context plumbing.

    For static typing, ``ContextOp[In, Out]`` should be read as if the
    operator were a normal ``In -> Out`` pipeline step. The ``Context``
    argument and returned ``Context`` are internal runtime details used by the
    pipeline executor to thread context updates through the operator chain.

    Most context operators can express their public boundary directly through
    the class declaration, for example ``Store(ContextOp[T, T])``. Operators
    whose output also depends on the incoming current shape, such as
    ``Recall``, can refine that public boundary with overloads on ``apply``.
    """

    @abstractmethod
    def apply(self, current: InputT, context: Context) -> tuple[OutputT, Context]:
        raise NotImplementedError

    @abstractmethod
    def resolve_contract(
        self,
        upstream_annotation: Any,
        stored_annotations: dict[str, Any],
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        raise NotImplementedError


@Operator
class Store(ContextOp[CurrentT, CurrentT]):
    def __init__(
        self,
        name: str,
        *,
        source: SelectorInput | None = None,
    ):
        self.name = name
        self._selector = Selector.from_input(source)

    def apply(self, current: CurrentT, context: Context) -> tuple[CurrentT, Context]:
        value = self._selector.select_value(
            current,
            error_prefix=f"Store({self.name!r}, {self._selector!r})",
        )
        return current, context.store(self.name, value)

    def resolve_contract(
        self,
        upstream_annotation: Any,
        stored_annotations: dict[str, Any],
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        stored_annotations[self.name] = self._selector.validate_read(
            upstream_annotation,
            validation_error_type=validation_error_type,
            error_prefix=f"Store({self.name!r}, {self._selector!r})",
        )
        return (Any,), upstream_annotation


@Operator
class Recall(ContextOp[Any, Any], Generic[StoredT, PrependT]):
    """Recall a stored value and append or prepend it to the current tuple shape.

    The public boundary depends on both the stored value type and the incoming
    ``current`` shape, so the class declaration stays broad and the ``apply``
    overloads carry the precise external typing.
    """

    @overload
    def __init__(self: "Recall[StoredT, Literal[False]]", name: str, prepend: Literal[False] = False) -> None:
        ...

    @overload
    def __init__(self: "Recall[StoredT, Literal[True]]", name: str, prepend: Literal[True]) -> None:
        ...

    def __init__(self, name: str, prepend: bool = False):
        self.name = name
        self.prepend = prepend

    @overload
    def apply(
        self: "Recall[StoredT, Literal[False]]",
        current: CurrentT,
        context: Context,
    ) -> tuple[tuple[CurrentT, StoredT], Context]:
        ...

    @overload
    def apply(
        self: "Recall[StoredT, Literal[True]]",
        current: CurrentT,
        context: Context,
    ) -> tuple[tuple[StoredT, CurrentT], Context]:
        ...

    def apply(self, current: CurrentT, context: Context) -> tuple[Any, Context]:
        stored = context.load(self.name)
        current_tuple = current if isinstance(current, tuple) else (current,)
        if self.prepend:
            result = (stored,) + current_tuple
        else:
            result = current_tuple + (stored,)
        return result, context

    def resolve_contract(
        self,
        upstream_annotation: Any,
        stored_annotations: dict[str, Any],
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del validation_error_type
        stored_annotation = stored_annotations.get(self.name, Any)
        current_item_annotation = variadic_tuple_item_annotation(upstream_annotation)
        if current_item_annotation is not None:
            merged_item_annotation = build_union_annotation_from_options(
                current_item_annotation,
                stored_annotation,
            )
            return (Any,), tuple[merged_item_annotation, ...]

        current_parts = expand_annotation_parts(upstream_annotation)
        if self.prepend:
            result_parts = (stored_annotation,) + current_parts
        else:
            result_parts = current_parts + (stored_annotation,)
        return (Any,), result_parts
