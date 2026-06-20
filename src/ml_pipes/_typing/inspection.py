from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .signatures import _POSITIONAL_VALUE_PARAMETER_KINDS


@dataclass(frozen=True)
class CallableAnnotations:
    parameter_annotations: tuple[Any | None, ...]
    return_annotation: Any | None


def resolve_callable_annotations(
    callable_: Callable[..., Any],
) -> CallableAnnotations:
    positional_parameters = _resolve_positional_value_parameters(callable_)
    if positional_parameters is None:
        return CallableAnnotations((), None)

    try:
        hints = _resolve_callable_hints(callable_)
    except (TypeError, ValueError):
        return CallableAnnotations(
            tuple(None for _ in positional_parameters),
            None,
        )

    return_annotation = callable_ if inspect.isclass(callable_) else hints.get("return")
    return CallableAnnotations(
        tuple(hints.get(parameter.name) for parameter in positional_parameters),
        return_annotation,
    )


def probe_callable(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> inspect.BoundArguments:
    return inspect.signature(callable_).bind(*args, **kwargs)


def _resolve_positional_value_parameters(
    callable_: Callable[..., Any],
) -> tuple[inspect.Parameter, ...] | None:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return None

    return tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
    )


def _resolve_callable_hints(callable_: Callable[..., Any]) -> dict[str, Any]:
    return get_type_hints(_resolve_callable_hints_target(callable_))


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
