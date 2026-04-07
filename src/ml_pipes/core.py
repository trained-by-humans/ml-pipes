from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Generic, Iterable, Mapping, Protocol, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")
PayloadInT = TypeVar("PayloadInT")
PayloadOutT = TypeVar("PayloadOutT")


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


class Operator(Protocol[PayloadInT, PayloadOutT]):
    def __call__(self, value: Value[PayloadInT]) -> Value[PayloadOutT]:
        ...


class PipelineValidationError(ValueError):
    pass


class Pipeline:
    def __init__(self, operators: Iterable[Operator[Any, Any]], validate_on_init: bool = False):
        self.operators = list(operators)
        if validate_on_init:
            self.validate()

    def __call__(self, value: Any) -> Any:
        current = value if isinstance(value, Value) else Value(value)
        for operator in self.operators:
            current = operator(current)
        return current

    def validate(self) -> None:
        if not self.operators:
            return

        previous_output_type: Any | None = None
        previous_name: str | None = None

        for operator in self.operators:
            input_type, output_type = self._resolve_operator_contract(operator)
            name = operator.__class__.__name__

            if previous_output_type is not None and not self._is_annotation_compatible(
                previous_output_type, input_type
            ):
                raise PipelineValidationError(
                    f"Pipeline contract mismatch: {previous_name} returns "
                    f"{self._format_annotation(previous_output_type)} but {name} expects "
                    f"{self._format_annotation(input_type)}"
                )

            previous_output_type = output_type
            previous_name = name

    @staticmethod
    def _resolve_operator_contract(operator: Operator[Any, Any]) -> tuple[Any, Any]:
        call = getattr(operator, "__call__")
        hints = get_type_hints(call)
        signature = inspect.signature(call)
        parameters = list(signature.parameters.values())

        if len(parameters) != 1:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} must define exactly one input parameter in __call__"
            )

        parameter = parameters[0]
        if parameter.name not in hints:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} is missing a type annotation for __call__ input"
            )
        if "return" not in hints:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} is missing a return type annotation for __call__"
            )

        return hints[parameter.name], hints["return"]

    @classmethod
    def _is_annotation_compatible(cls, produced: Any, expected: Any) -> bool:
        if expected is Any or produced is Any:
            return True
        if produced == expected:
            return True

        if cls._is_concrete_assignable(produced, expected):
            return True

        produced_origin = get_origin(produced)
        expected_origin = get_origin(expected)

        if cls._is_union_annotation(expected):
            return any(cls._is_annotation_compatible(produced, option) for option in get_args(expected))
        if cls._is_union_annotation(produced):
            return all(cls._is_annotation_compatible(option, expected) for option in get_args(produced))

        if produced_origin is None or expected_origin is None:
            return False
        if produced_origin != expected_origin:
            return False

        produced_args = get_args(produced)
        expected_args = get_args(expected)
        if len(produced_args) != len(expected_args):
            return False

        return all(
            cls._is_annotation_compatible(produced_arg, expected_arg)
            for produced_arg, expected_arg in zip(produced_args, expected_args, strict=True)
        )

    @staticmethod
    def _is_concrete_assignable(produced: Any, expected: Any) -> bool:
        if not isinstance(produced, type) or not isinstance(expected, type):
            return False
        try:
            return issubclass(produced, expected)
        except TypeError:
            return False

    @staticmethod
    def _is_union_annotation(annotation: Any) -> bool:
        origin = get_origin(annotation)
        return origin in (UnionType, getattr(__import__("typing"), "Union"))

    @staticmethod
    def _format_annotation(annotation: Any) -> str:
        return str(annotation).replace("typing.", "")
