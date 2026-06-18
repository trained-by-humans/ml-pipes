from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Mapping, TypeVar, overload

from ._typing.annotation import combine_annotation_options, variadic_tuple_item_annotation
from .operator import Operator
from .selector import Selector, SelectorInput

CurrentT = TypeVar("CurrentT")
InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
StoredT = TypeVar("StoredT")
InsertIndexT = TypeVar("InsertIndexT", bound=int | None)


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
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
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
class Recall(ContextOp[Any, Any], Generic[StoredT, InsertIndexT]):
    """Recall a stored value and splice it into the current tuple shape.

    The public boundary depends on both the stored value type and the incoming
    ``current`` shape, so the class declaration stays broad and the ``apply``
    overloads carry the precise external typing.
    """

    @overload
    def __init__(self: "Recall[StoredT, None]", name: str, index: None = None) -> None:
        ...

    @overload
    def __init__(self: "Recall[StoredT, Literal[0]]", name: str, index: Literal[0]) -> None:
        ...

    @overload
    def __init__(self: "Recall[StoredT, int]", name: str, index: int) -> None:
        ...

    def __init__(self, name: str, index: int | None = None):
        self.name = name
        self.index = index

    @overload
    def apply(
        self: "Recall[StoredT, None]",
        current: CurrentT,
        context: Context,
    ) -> tuple[tuple[CurrentT, StoredT], Context]:
        ...

    @overload
    def apply(
        self: "Recall[StoredT, Literal[0]]",
        current: CurrentT,
        context: Context,
    ) -> tuple[tuple[StoredT, CurrentT], Context]:
        ...

    @overload
    def apply(
        self: "Recall[StoredT, int]",
        current: CurrentT,
        context: Context,
    ) -> tuple[Any, Context]:
        ...

    def apply(self, current: CurrentT, context: Context) -> tuple[Any, Context]:
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
        current_item_annotation = variadic_tuple_item_annotation(current_output)
        if current_item_annotation is not None:
            merged_item_annotation = combine_annotation_options(
                current_item_annotation,
                stored_annotation,
            )
            return (Any,), tuple[merged_item_annotation, ...]

        current_parts = expand_output_annotation(current_output)
        if self.index is None:
            result_parts = current_parts + (stored_annotation,)
        else:
            result_parts = current_parts[:self.index] + (stored_annotation,) + current_parts[self.index:]
        return (Any,), result_parts
