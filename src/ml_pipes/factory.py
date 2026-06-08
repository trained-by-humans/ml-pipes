from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Generic, TypeVar

_PIPELINE_FACTORY_ATTR = "_ml_pipes_pipeline_factory"
_DATA_FACTORY_ATTR = "_ml_pipes_data_factory"
FactoryOutputT = TypeVar("FactoryOutputT")

# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


class Factory(Generic[FactoryOutputT]):
    """Callable wrapper with a public config-driven factory entrypoint."""

    def __init__(
        self,
        fn: Callable[..., FactoryOutputT],
        *,
        from_config: Callable[[dict], FactoryOutputT],
        attr: str | None = None,
        signature_target: Callable | None = None,
    ) -> None:
        self._fn = fn
        self._from_config = from_config
        self._signature_target = signature_target
        functools.update_wrapper(self, fn)
        if attr is not None:
            setattr(self, attr, True)

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., FactoryOutputT] | Factory[FactoryOutputT],
        *,
        attr: str | None = None,
    ) -> Factory[FactoryOutputT]:
        """Adapt ``fn(**config)`` into the ``Factory`` interface.

        Idempotent for existing ``Factory`` instances.
        """
        if isinstance(fn, cls):
            if attr is not None:
                setattr(fn, attr, True)
            return fn
        return cls(fn, from_config=_wrap_as_factory(fn), attr=attr, signature_target=fn)

    @classmethod
    def from_config_callable(
        cls,
        fn: Callable[[dict], FactoryOutputT] | Factory[FactoryOutputT],
        *,
        attr: str | None = None,
    ) -> Factory[FactoryOutputT]:
        """Adapt ``fn(config_dict)`` into the ``Factory`` interface.

        Idempotent for existing ``Factory`` instances.
        """
        if isinstance(fn, cls):
            if attr is not None:
                setattr(fn, attr, True)
            return fn
        return cls(fn, from_config=fn, attr=attr, signature_target=None)

    @property
    def signature_target(self) -> Callable | None:
        return self._signature_target

    def from_config(self, config: dict) -> FactoryOutputT:
        return self._from_config(config)

    def __call__(self, *args, **kwargs) -> FactoryOutputT:
        return self._fn(*args, **kwargs)


def pipeline_factory(fn: Callable) -> Callable:
    """Mark a function as a pipeline factory for CLI discovery.

    The decorated value is a ``Factory`` object. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. Any parameter without a default must be supplied
    through ``--arg``, ``--config``, or ``--axis``.
    """
    return Factory.from_callable(fn, attr=_PIPELINE_FACTORY_ATTR)


def data_factory(fn: Callable) -> Callable:
    """Mark a function as a data factory for CLI discovery.

    The decorated value is a ``Factory`` object. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. It must return an ``InputFn`` — a zero-argument
    callable yielding ``(id: str, value: Any, tag: str | None,
    metadata: dict | None)``.
    """
    return Factory.from_callable(fn, attr=_DATA_FACTORY_ATTR)


def _wrap_as_factory(fn: Callable) -> Callable:
    """Wrap a plain callable so it accepts (config: dict) and calls fn(**config)."""
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    return wrapper


def _signature_target(factory: Callable) -> Callable | None:
    if isinstance(factory, Factory):
        return factory.signature_target
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
