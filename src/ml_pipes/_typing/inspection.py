from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .signatures import _POSITIONAL_VALUE_PARAMETER_KINDS


@dataclass(frozen=True)
class CallableAnnotations:
    parameter_annotations: tuple[Any | None, ...]
    return_annotation: Any | None


def probe_callable(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> inspect.BoundArguments:
    return inspect.signature(callable_).bind(*args, **kwargs)


def _resolve_callable_hints_and_return_annotation(
    callable_: Callable[..., Any],
) -> tuple[dict[str, Any], Any | None]:
    hints = resolve_callable_hints(callable_)
    return_annotation = callable_ if inspect.isclass(callable_) else hints.get("return")
    return hints, return_annotation


def resolve_callable_hints(callable_: Callable[..., Any]) -> dict[str, Any]:
    return get_type_hints(_resolve_callable_hints_target(callable_))


def resolve_unary_callable_annotations(
    callable_: Callable[..., Any],
) -> CallableAnnotations:
    input_parameter = _resolve_first_positional_value_parameter(callable_)
    if input_parameter is None:
        return CallableAnnotations((None,), None)

    try:
        hints, return_annotation = _resolve_callable_hints_and_return_annotation(callable_)
    except (TypeError, ValueError):
        return CallableAnnotations((None,), None)
    return CallableAnnotations((hints.get(input_parameter.name),), return_annotation)


def resolve_nullary_callable_annotations(
    callable_: Callable[..., Any],
) -> CallableAnnotations:
    try:
        _, return_annotation = _resolve_callable_hints_and_return_annotation(callable_)
    except (TypeError, ValueError):
        return CallableAnnotations((), None)
    return CallableAnnotations((), return_annotation)


def _resolve_first_positional_value_parameter(
    callable_: Callable[..., Any],
) -> inspect.Parameter | None:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return None

    return next(
        (
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
        ),
        None,
    )


def _resolve_callable_hints_target(callable_: Callable[..., Any]) -> Any:
    if inspect.isclass(callable_):
        return getattr(callable_, "__init__", callable_)
    if (
        inspect.isfunction(callable_)
        or inspect.ismethod(callable_)
        or inspect.isbuiltin(callable_)
        or inspect.ismethoddescriptor(callable_)
    ):
        return callable_
    return getattr(callable_, "__call__", callable_)
