from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Generic, TypeVar

from .core import Pipeline

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
        fn: Callable[..., FactoryOutputT],
        *,
        attr: str | None = None,
    ) -> Factory[FactoryOutputT]:
        """Build a ``Factory`` from a keyword-style callable."""
        if isinstance(fn, cls):
            raise TypeError("Factory.from_callable() expects a plain callable; use Factory.ensure_factory().")
        return cls(fn, from_config=_wrap_as_factory(fn), attr=attr, signature_target=fn)

    @classmethod
    def from_config_callable(
        cls,
        fn: Callable[[dict], FactoryOutputT],
        *,
        attr: str | None = None,
    ) -> Factory[FactoryOutputT]:
        """Build a ``Factory`` from a config-dict callable."""
        if isinstance(fn, cls):
            raise TypeError(
                "Factory.from_config_callable() expects a config callable; use Factory.ensure_factory()."
            )
        return cls(fn, from_config=fn, attr=attr, signature_target=None)

    @classmethod
    def ensure_factory(
        cls,
        fn: Callable[[dict], FactoryOutputT] | Factory[FactoryOutputT],
    ) -> Factory[FactoryOutputT]:
        """Normalize a config-dict callable or existing ``Factory`` to ``Factory``."""
        if isinstance(fn, cls):
            return fn
        return cls.from_config_callable(fn)

    @property
    def signature_target(self) -> Callable | None:
        return self._signature_target

    def validate_config(self, config: dict, *, name: str = "factory") -> None:
        """Validate a config dict against the wrapped keyword signature."""
        target = self.signature_target
        if target is None:
            return

        params = inspect.signature(target).parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not has_var_keyword:
            unexpected = [k for k in config if k not in params]
            if unexpected:
                raise TypeError(
                    f"{name} got unknown config key(s) {unexpected!r} for config {config!r}"
                )

        missing = [
            param_name
            for param_name, param in params.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and param_name not in config
        ]
        if missing:
            raise TypeError(
                f"{name} is missing required config key(s) {missing!r} for config {config!r}"
            )

    def from_config(self, config: dict) -> FactoryOutputT:
        return self._from_config(config)

    def __call__(self, *args, **kwargs) -> FactoryOutputT:
        return self._fn(*args, **kwargs)

    @staticmethod
    def discover_pipeline(module: Any, explicit_fn: Any = None) -> Factory[Pipeline] | None:
        """Discover the module's pipeline factory, normalized to ``Factory``."""
        return Factory._discover(module, explicit_fn, attr=_PIPELINE_FACTORY_ATTR, kind="pipeline")

    @staticmethod
    def discover_data(module: Any, explicit_fn: Any = None) -> Factory[InputFn] | None:
        """Discover the module's data factory, normalized to ``Factory``."""
        return Factory._discover(module, explicit_fn, attr=_DATA_FACTORY_ATTR, kind="data")

    @staticmethod
    def _discover(
        module: Any,
        explicit_fn: Any,
        *,
        attr: str,
        kind: str,
    ) -> Factory[Any] | None:
        if explicit_fn is not None:
            if isinstance(explicit_fn, Factory):
                return explicit_fn
            if getattr(explicit_fn, attr, False):
                return Factory.from_callable(explicit_fn, attr=attr)
            return Factory.from_config_callable(explicit_fn)

        found = _discover_marked_factory(module, attr, kind)
        if found is None:
            return None
        if isinstance(found, Factory):
            return found
        return Factory.from_callable(found, attr=attr)


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


def _discover_marked_factory(module: Any, attr: str, kind: str) -> Any:
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
