from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

_PIPELINE_FACTORY_ATTR = "_ml_pipes_pipeline_factory"
_DATA_FACTORY_ATTR = "_ml_pipes_data_factory"

# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


def pipeline_factory(fn: Callable) -> Callable:
    """Mark a function as a pipeline factory for CLI discovery.

    The decorated function may have any signature; the CLI calls it via
    ``factory(config_dict)`` which unpacks to ``fn(**config)``.  Any parameter
    without a default must be supplied through ``--arg``, ``--config``, or ``--axis``.
    """
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    setattr(wrapper, _PIPELINE_FACTORY_ATTR, True)
    return wrapper


def data_factory(fn: Callable) -> Callable:
    """Mark a function as a data factory for CLI discovery.

    The decorated function may have any signature and must return an
    ``InputFn`` — a zero-argument callable yielding
    ``(id: str, value: Any, tag: str | None, metadata: dict | None)``.
    """
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    setattr(wrapper, _DATA_FACTORY_ATTR, True)
    return wrapper


def _wrap_as_factory(fn: Callable) -> Callable:
    """Wrap a plain callable so it accepts (config: dict) and calls fn(**config)."""
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    return wrapper


def discover_factory(module: Any, explicit_fn: Any, attr: str, kind: str) -> Any:
    """Scan module for a function marked with attr, or return explicit_fn if given.

    Explicit refs that are not already decorated are wrapped to accept (config: dict)
    so they behave identically to @pipeline_factory / @data_factory callables.

    Returns None when nothing is found.
    Raises ValueError when more than one decorated function is present.
    """
    if explicit_fn is not None:
        if not getattr(explicit_fn, attr, False):
            return _wrap_as_factory(explicit_fn)
        return explicit_fn

    found = [
        (name, fn)
        for name, fn in vars(module).items()
        if callable(fn) and getattr(fn, attr, False)
    ]

    if len(found) > 1:
        names = ", ".join(name for name, _ in found)
        raise ValueError(
            f"multiple @{kind}_factory found in {module.__name__!r}: [{names}]. "
            f"Only one is allowed per module; remove one or use 'module:{kind}_factory_fn' syntax."
        )

    return found[0][1] if found else None


def validate_factory_config(factory: Callable, config: dict) -> None:
    """Raise TypeError if config is missing required parameters for factory.

    Inspects the original function via __wrapped__ so decorated factories
    with natural keyword signatures are validated correctly.
    """
    wrapped = getattr(factory, "__wrapped__", None)
    if wrapped is None:
        return
    inspect.signature(wrapped).bind(**config)
