from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable


_POSITIONAL_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
_POSITIONAL_VALUE_PARAMETER_KINDS = _POSITIONAL_PARAMETER_KINDS + (
    inspect.Parameter.VAR_POSITIONAL,
)


def validate_operator_signature(
    operator: Any,
    *,
    label: str,
    error_type: type[Exception],
    warning_type: type[Warning] | None = None,
) -> tuple[inspect.Parameter, ...]:
    try:
        signature = inspect.signature(operator)
    except (TypeError, ValueError) as exc:
        if not callable(operator):
            raise error_type(
                f"Pipeline step {label} must define __call__"
            ) from exc
        raise error_type(
            f"Pipeline step {label} must expose an inspectable call signature"
        ) from exc
    parameters = tuple(signature.parameters.values())
    variadic_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL
    )
    if variadic_parameters:
        raise error_type(
            f"Pipeline step {label} uses variadic positional parameters "
            f"({_format_parameter_descriptions(variadic_parameters)}), "
            f"which Pipeline does not support. Use a single tuple-typed parameter instead."
        )

    unsupported_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind not in _POSITIONAL_VALUE_PARAMETER_KINDS
    )
    if unsupported_parameters:
        raise error_type(
            f"Pipeline step {label} uses non-positional parameters "
            f"({_format_parameter_descriptions(unsupported_parameters)}), but Pipeline chains operators by "
            f"argument position. Use only positional parameters in __call__."
        )

    if not parameters:
        raise error_type(
            f"Pipeline step {label} must define at least one positional input parameter in __call__"
        )

    if warning_type is not None and len(parameters) > 1:
        defaulted_parameters = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
            and parameter.default is not inspect.Parameter.empty
        )
        if defaulted_parameters:
            defaulted_parameter_names = ", ".join(
                parameter.name for parameter in defaulted_parameters
            )
            warnings.warn(
                f"Pipeline step {label} defines positional defaults ({defaulted_parameter_names}), "
                f"but Pipeline ignores positional defaults for dispatch to avoid ambiguity with tuple-valued "
                f"pipeline outputs. Validation will treat this operator as requiring "
                f"{len(parameters)} positional pipeline inputs.",
                warning_type,
                stacklevel=4,
            )

    return parameters


def _validate_positional_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    source_label: str,
    error_type: type[Exception],
) -> inspect.Parameter:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} must expose an inspectable call signature because {source_label} is passed by position"
        ) from exc
    parameters = tuple(signature.parameters.values())

    input_parameter = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
        ),
        None,
    )
    if input_parameter is None:
        raise error_type(
            f"{label} must accept at least one positional value from {source_label}"
        )

    unsupported_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind not in _POSITIONAL_VALUE_PARAMETER_KINDS
    )
    if unsupported_parameters:
        raise error_type(
            f"{label} uses non-positional parameters "
            f"({_format_parameter_descriptions(unsupported_parameters)}), "
            f"but {source_label} is passed by position. Use only positional parameters."
        )

    return input_parameter


def validate_unary_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    source_label: str,
    error_type: type[Exception],
) -> inspect.Parameter:
    input_parameter = _validate_positional_callable_signature(
        callable_,
        label=label,
        source_label=source_label,
        error_type=error_type,
    )
    try:
        inspect.signature(callable_).bind(object())
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} cannot be called with {source_label}: {exc}"
        ) from exc
    return input_parameter


def validate_nullary_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    source_label: str,
    error_type: type[Exception],
) -> None:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} must expose an inspectable call signature because {source_label}"
        ) from exc
    parameters = tuple(signature.parameters.values())
    if parameters:
        raise error_type(
            f"{label} must define no parameters because {source_label}, but declares "
            f"({_format_parameter_descriptions(parameters)}). "
            f"Use a zero-argument callable with any configuration pre-bound."
        )


def _format_parameter_descriptions(parameters: tuple[inspect.Parameter, ...]) -> str:
    return ", ".join(
        f"{parameter.name} ({parameter.kind.name.lower().replace('_', ' ')})"
        for parameter in parameters
    )
