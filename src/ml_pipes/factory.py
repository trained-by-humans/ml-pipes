from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Generic, TypeVar

from .core import Pipeline

FactoryOutputT = TypeVar("FactoryOutputT")
FactoryClassT = TypeVar("FactoryClassT", bound="Factory[Any]")

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
        signature_target: Callable | None = None,
    ) -> None:
        self._fn = fn
        self._from_config = from_config
        self._signature_target = signature_target
        functools.update_wrapper(self, fn)

    @classmethod
    def from_callable(
        cls: type[FactoryClassT],
        fn: Callable[..., FactoryOutputT],
    ) -> FactoryClassT:
        """Build a ``Factory`` from a keyword-style callable."""
        if isinstance(fn, Factory):
            raise TypeError(
                f"{cls.__name__}.from_callable() expects a plain callable; use {cls.__name__}.ensure_factory()."
            )
        return cls(fn, from_config=_wrap_as_factory(fn), signature_target=fn)

    @classmethod
    def from_config_callable(
        cls: type[FactoryClassT],
        fn: Callable[[dict], FactoryOutputT],
    ) -> FactoryClassT:
        """Build a ``Factory`` from a config-dict callable."""
        if isinstance(fn, Factory):
            raise TypeError(
                f"{cls.__name__}.from_config_callable() expects a config callable; "
                f"use {cls.__name__}.ensure_factory()."
            )
        return cls(fn, from_config=fn, signature_target=None)

    @classmethod
    def ensure_factory(
        cls: type[FactoryClassT],
        fn: Callable[[dict], FactoryOutputT] | Factory[Any],
    ) -> FactoryClassT:
        """Normalize a config-dict callable or existing compatible factory."""
        if isinstance(fn, cls):
            return fn
        if isinstance(fn, Factory):
            if type(fn) is Factory:
                return cls(fn._fn, from_config=fn._from_config, signature_target=fn.signature_target)
            raise TypeError(
                f"{cls.__name__}.ensure_factory() expects a config callable or {cls.__name__}, "
                f"got {type(fn).__name__}."
            )
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

    @classmethod
    def _discover(
        cls: type[FactoryClassT],
        module: Any,
        explicit_fn: Any,
    ) -> FactoryClassT | None:
        if explicit_fn is not None:
            return cls.ensure_factory(explicit_fn)

        found = _discover_marked_factory(module, cls)
        if found is None:
            return None
        return found


class PipelineFactory(Factory[Pipeline]):
    """Factory subtype for pipeline-producing callables."""

    @classmethod
    def discover(cls, module: Any, explicit_fn: Any = None) -> PipelineFactory | None:
        """Discover the module's pipeline factory, normalized to ``PipelineFactory``."""
        return cls._discover(module, explicit_fn)


class DataFactory(Factory[InputFn]):
    """Factory subtype for input-producing callables."""

    @classmethod
    def discover(cls, module: Any, explicit_fn: Any = None) -> DataFactory | None:
        """Discover the module's data factory, normalized to ``DataFactory``."""
        return cls._discover(module, explicit_fn)


def pipeline_factory(fn: Callable[..., Pipeline]) -> PipelineFactory:
    """Mark a function as a pipeline factory for CLI discovery.

    The decorated value is a ``PipelineFactory``. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. Any parameter without a default must be supplied
    through ``--arg``, ``--config``, or ``--axis``.
    """
    return PipelineFactory.from_callable(fn)


def data_factory(fn: Callable[..., InputFn]) -> DataFactory:
    """Mark a function as a data factory for CLI discovery.

    The decorated value is a ``DataFactory``. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. It must return an ``InputFn`` — a zero-argument
    callable yielding ``(id: str, value: Any, tag: str | None,
    metadata: dict | None)``.
    """
    return DataFactory.from_callable(fn)


def _wrap_as_factory(fn: Callable) -> Callable:
    """Wrap a plain callable so it accepts (config: dict) and calls fn(**config)."""
    @functools.wraps(fn)
    def wrapper(config: dict) -> Any:
        return fn(**config)
    return wrapper


def _discover_marked_factory(
    module: Any,
    factory_type: type[FactoryClassT],
) -> FactoryClassT | None:
    kind = _factory_kind_name(factory_type)
    found = [
        (name, fn)
        for name, fn in vars(module).items()
        if isinstance(fn, factory_type)
    ]

    if len(found) > 1:
        names = ", ".join(name for name, _ in found)
        raise ValueError(
            f"multiple @{kind}_factory found in {module.__name__!r}: [{names}]. "
            f"Only one is allowed per module; remove one or use 'module:{kind}_factory_fn' syntax."
        )

    return found[0][1] if found else None


def _factory_kind_name(factory_type: type[Factory[Any]]) -> str:
    return factory_type.__name__.removesuffix("Factory").lower()
