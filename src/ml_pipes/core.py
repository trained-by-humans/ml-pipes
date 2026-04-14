from __future__ import annotations

import inspect
from types import UnionType
from typing import Any, Callable, Iterable, get_args, get_origin, get_type_hints

from .context import Context, ContextOp


class PipelineValidationError(ValueError):
    pass


class Pipeline:
    def __init__(self, operators: Iterable[Callable[..., Any] | ContextOp], validate_on_init: bool = False):
        self.operators = list(operators)
        if validate_on_init:
            self.validate()

    def __call__(self, value: Any) -> Any:
        current = value
        context = Context()
        for operator in self.operators:
            if isinstance(operator, ContextOp):
                current, context = operator.apply(current, context)
            else:
                args = self._build_call_args(operator, current)
                current = operator(*args)
        return current

    def validate(self) -> None:
        if not self.operators:
            return

        previous_output_type: Any | None = None
        previous_name: str | None = None
        stored_annotations: dict[str, Any] = {}

        for operator in self.operators:
            if isinstance(operator, ContextOp) or hasattr(operator, "resolve_contract"):
                _, output_type = operator.resolve_contract(
                    previous_output_type,
                    stored_annotations,
                    self._expand_output_annotation,
                    PipelineValidationError,
                )
            else:
                input_types, output_type = self._resolve_operator_contract(operator)
                name = operator.__class__.__name__

                if previous_output_type is not None and not self._is_annotation_compatible(
                    previous_output_type, input_types
                ):
                    raise PipelineValidationError(
                        f"Pipeline contract mismatch: {previous_name} returns "
                        f"{self._format_annotation(previous_output_type)} but {name} expects "
                        f"{self._format_parameter_annotations(input_types)}"
                    )

            previous_output_type = output_type
            previous_name = operator.__class__.__name__

    @staticmethod
    def _resolve_operator_contract(operator: Callable[..., Any]) -> tuple[tuple[Any, ...], Any]:
        target = Pipeline._get_signature_target(operator)
        hints = get_type_hints(target)
        signature = inspect.signature(target)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if not parameters:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} must define at least one positional input parameter in __call__"
            )

        input_types: list[Any] = []
        for parameter in parameters:
            if parameter.name not in hints:
                raise PipelineValidationError(
                    f"{operator.__class__.__name__} is missing a type annotation for __call__ input"
                )
            input_types.append(hints[parameter.name])
        if "return" not in hints:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} is missing a return type annotation for __call__"
            )

        return tuple(input_types), hints["return"]

    @classmethod
    def _is_annotation_compatible(cls, produced: Any, expected_inputs: tuple[Any, ...]) -> bool:
        # When the next operator takes a single parameter, _build_call_args passes
        # the whole current value without unpacking — so compare the unexpanded
        # produced type against that single expected type.
        if len(expected_inputs) == 1:
            return cls._is_single_annotation_compatible(produced, expected_inputs[0])

        produced_types = cls._expand_output_annotation(produced)
        if len(produced_types) != len(expected_inputs):
            return False

        return all(
            cls._is_single_annotation_compatible(produced_type, expected_type)
            for produced_type, expected_type in zip(produced_types, expected_inputs, strict=True)
        )

    @classmethod
    def _is_single_annotation_compatible(cls, produced: Any, expected: Any) -> bool:
        if expected is Any or produced is Any:
            return True
        if produced == expected:
            return True

        if cls._is_concrete_assignable(produced, expected):
            return True

        produced_origin = get_origin(produced)
        expected_origin = get_origin(expected)

        if cls._is_union_annotation(expected):
            return any(cls._is_single_annotation_compatible(produced, option) for option in get_args(expected))
        if cls._is_union_annotation(produced):
            return all(cls._is_single_annotation_compatible(option, expected) for option in get_args(produced))

        if produced_origin is None:
            return False
        if expected_origin is None:
            # e.g. tuple[int, str] is assignable to bare tuple
            return cls._is_concrete_assignable(produced_origin, expected)
        if produced_origin != expected_origin:
            return False

        produced_args = get_args(produced)
        expected_args = get_args(expected)
        if len(produced_args) != len(expected_args):
            return False

        return all(
            cls._is_single_annotation_compatible(produced_arg, expected_arg)
            for produced_arg, expected_arg in zip(produced_args, expected_args, strict=True)
        )

    @staticmethod
    def _expand_output_annotation(annotation: Any) -> tuple[Any, ...]:
        origin = get_origin(annotation)
        if origin is tuple:
            return get_args(annotation)
        if isinstance(annotation, tuple):
            return annotation
        return (annotation,)

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

    @classmethod
    def _format_parameter_annotations(cls, annotations: tuple[Any, ...]) -> str:
        if len(annotations) == 1:
            return cls._format_annotation(annotations[0])
        return "(" + ", ".join(cls._format_annotation(annotation) for annotation in annotations) + ")"

    @staticmethod
    def _build_call_args(operator: Callable[..., Any], current: Any) -> tuple[Any, ...]:
        signature = inspect.signature(Pipeline._get_signature_target(operator))
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if len(parameters) == 1:
            return (current,)
        if isinstance(current, tuple):
            if len(current) != len(parameters):
                raise TypeError(
                    f"{operator.__class__.__name__} expects {len(parameters)} positional arguments, "
                    f"got tuple of length {len(current)}"
                )
            return current
        raise TypeError(
            f"{operator.__class__.__name__} expects {len(parameters)} positional arguments, got 1"
        )

    @staticmethod
    def _get_signature_target(operator: Callable[..., Any]) -> Any:
        if inspect.isfunction(operator) or inspect.ismethod(operator):
            return operator
        return getattr(operator, "__call__")
