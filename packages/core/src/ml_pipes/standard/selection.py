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
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        if current_output is Any:
            return (Any,), Any

        selected = self._selector.validate_read(
            current_output,
            validation_error_type=validation_error_type,
            error_prefix=f"Select({self._selector!r})",
        )
        return (current_output,), selected


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
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation

        if current_output is Any or current_output is tuple:
            return (tuple[Any, ...],), Any

        repeated_item = self._homogeneous_tuple_item(current_output)
        if repeated_item is not None:
            selected = tuple(repeated_item for _ in self.indices)
            return (current_output,), selected[0] if len(selected) == 1 else tuple[selected]

        parts = self._fixed_tuple_parts(current_output)
        if parts is None:
            raise validation_error_type(f"Pick requires a tuple boundary, got {current_output}")

        selected = tuple(
            parts[
                self._normalize_fixed_index(
                    index,
                    len(parts),
                    current_output,
                    validation_error_type,
                )
            ]
            for index in self.indices
        )
        input_annotation = current_output if get_origin(current_output) is tuple else tuple[parts]
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
        current_output: Any,
        error_type: type[Exception],
    ) -> int:
        normalized_index = index if index >= 0 else size + index
        if normalized_index < 0 or normalized_index >= size:
            raise error_type(
                f"Pick({index}) is out of bounds for {current_output} (length {size})"
            )
        return normalized_index
