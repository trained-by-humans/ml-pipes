from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar, get_args, get_origin, overload

from ml_pipes.operator import Operator
from ml_pipes.selector import Selector, SelectorInput

PickIndexT = TypeVar("PickIndexT", bound=int)
PickFirstT = TypeVar("PickFirstT")
PickSecondT = TypeVar("PickSecondT")


@Operator
class Select:
    def __init__(self, *path: SelectorInput):
        self._selector = Selector.from_input(path)
        if not self._selector:
            raise ValueError("Select requires at least one selector part")

    def __call__(self, current: Any) -> Any:
        return self._selector.select_value(
            current,
            error_prefix=f"Select({self._selector!r})",
        )

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if upstream_annotation is Any:
            return (Any,), Any

        selected = self._selector.validate_read(
            upstream_annotation,
            validation_error_type=validation_error_type,
            error_prefix=f"Select({self._selector!r})",
        )
        return (upstream_annotation,), selected


@Operator
class Pick(Generic[PickIndexT]):
    @overload
    def __init__(self: "Pick[Literal[0]]", index: Literal[0]) -> None:
        ...

    @overload
    def __init__(self: "Pick[Literal[1]]", index: Literal[1]) -> None:
        ...

    @overload
    def __init__(self: "Pick[int]", *indices: int) -> None:
        ...

    def __init__(self, *indices: int):
        if not indices:
            raise ValueError("Pick requires at least one index")
        self.indices = indices

    @overload
    def __call__(
        self: "Pick[Literal[0]]",
        current: tuple[PickFirstT, PickSecondT],
    ) -> PickFirstT:
        ...

    @overload
    def __call__(
        self: "Pick[Literal[1]]",
        current: tuple[PickFirstT, PickSecondT],
    ) -> PickSecondT:
        ...

    @overload
    def __call__(self, current: tuple[Any, ...]) -> Any:
        ...

    def __call__(self, current: tuple[Any, ...]) -> Any:
        if not isinstance(current, tuple):
            raise TypeError("Pick can only be applied to tuple outputs")
        selected = tuple(current[index] for index in self.indices)
        if len(selected) == 1:
            return selected[0]
        return selected

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if upstream_annotation is Any or upstream_annotation is tuple:
            return (tuple[Any, ...],), Any

        repeated_item = self._homogeneous_tuple_item(upstream_annotation)
        if repeated_item is not None:
            selected = tuple(repeated_item for _ in self.indices)
            return (upstream_annotation,), selected[0] if len(selected) == 1 else tuple[selected]

        parts = self._fixed_tuple_parts(upstream_annotation)
        if parts is None:
            raise validation_error_type(f"Pick requires a tuple boundary, got {upstream_annotation}")

        selected = tuple(
            parts[
                self._normalize_fixed_index(
                    index,
                    len(parts),
                    upstream_annotation,
                    validation_error_type,
                )
            ]
            for index in self.indices
        )
        input_annotation = upstream_annotation if get_origin(upstream_annotation) is tuple else tuple[parts]
        return (input_annotation,), selected[0] if len(selected) == 1 else tuple[selected]

    @staticmethod
    def _fixed_tuple_parts(annotation: Any) -> tuple[Any, ...] | None:
        if isinstance(annotation, tuple):
            return annotation
        if get_origin(annotation) is not tuple:
            return None
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return None
        return args

    @staticmethod
    def _homogeneous_tuple_item(annotation: Any) -> Any | None:
        if get_origin(annotation) is not tuple:
            return None
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        return None

    @staticmethod
    def _normalize_fixed_index(
        index: int,
        size: int,
        upstream_annotation: Any,
        error_type: type[Exception],
    ) -> int:
        normalized_index = index if index >= 0 else size + index
        if normalized_index < 0 or normalized_index >= size:
            raise error_type(
                f"Pick({index}) is out of bounds for {upstream_annotation} (length {size})"
            )
        return normalized_index
