from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

_PIPELINE_FACTORY_ATTR = "_ml_pipes_pipeline_factory"
_DATA_FACTORY_ATTR = "_ml_pipes_data_factory"
_FACTORY_ADAPTER_ATTR = "_ml_pipes_factory_adapter"

# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


def pipeline_factory(fn: Callable) -> Callable:
    """Mark a function as a pipeline factory for CLI discovery.

    The decorated function keeps its original Python call semantics. CLI and
    benchmark helpers adapt it to ``factory(config_dict)`` and unpack to
    ``fn(**config)``. Any parameter without a default must be supplied through
    ``--arg``, ``--config``, or ``--axis``.
    """
    setattr(fn, _PIPELINE_FACTORY_ATTR, True)
    return fn


def data_factory(fn: Callable) -> Callable:
    """Mark a function as a data factory for CLI discovery.

    The decorated function keeps its original Python call semantics. CLI and
    benchmark helpers adapt it to config-dict invocation. It must return an
    ``InputFn`` — a zero-argument callable yielding ``(id: str, value: Any,
    tag: str | None, metadata: dict | None)``.
    """
    setattr(fn, _DATA_FACTORY_ATTR, True)
    return fn


def _wrap_as_factory(fn: Callable) -> Callable:
    """Wrap a plain callable so it accepts (config: dict) and calls fn(**config)."""
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    setattr(wrapper, _FACTORY_ADAPTER_ATTR, True)
    return wrapper


def coerce_factory(fn: Callable) -> Callable:
    """Adapt any callable to ``factory(config_dict)`` form via ``fn(**config)``.

    Idempotent: passing an already adapted factory returns it unchanged.
    """
    if getattr(fn, _FACTORY_ADAPTER_ATTR, False):
        return fn
    return _wrap_as_factory(fn)


def _signature_target(factory: Callable) -> Callable | None:
    return getattr(factory, "__wrapped__", None)


def discover_factory(module: Any, explicit_fn: Any, attr: str, kind: str) -> Any:
    """Scan module for a function marked with attr, or return explicit_fn if given.

    Returns None when nothing is found.
    Raises ValueError when more than one decorated function is present.
    """
    if explicit_fn is not None:
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

    Inspects the original function behind an adapted factory wrapper.
    """
    target = _signature_target(factory)
    if target is None:
        return
    inspect.signature(target).bind(**config)
