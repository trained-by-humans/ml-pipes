from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints


def probe_callable(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> inspect.BoundArguments:
    return inspect.signature(callable_).bind(*args, **kwargs)


def resolve_callable_annotations(
    callable_: Callable[..., Any],
) -> tuple[dict[str, Any], Any | None]:
    hints = resolve_callable_hints(callable_)
    output_annotation = callable_ if inspect.isclass(callable_) else hints.get("return")
    return hints, output_annotation


def resolve_callable_hints(callable_: Callable[..., Any]) -> dict[str, Any]:
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
