from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Generic, TypeVar

from .core import Pipeline

FactoryOutputT = TypeVar("FactoryOutputT")
FactoryT = TypeVar("FactoryT", bound="Factory[Any]")

# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


class Factory(Generic[FactoryOutputT]):
    """Callable wrapper with a public config-driven factory entrypoint."""

    _factory_name = "factory"

    def __init__(
        self,
        fn: Callable[..., FactoryOutputT],
        *,
        from_config: Callable[[dict], FactoryOutputT],
        signature_source: Callable | None = None,
    ) -> None:
        self._fn = fn
        self._from_config = from_config
        self._signature_source = signature_source
        functools.update_wrapper(self, fn)

    @classmethod
    def from_callable(
        cls: type[FactoryT],
        fn: Callable[..., FactoryOutputT],
    ) -> FactoryT:
        """Build a ``Factory`` from a plain callable and derive its config adapter."""
        if isinstance(fn, Factory):
            raise TypeError(
                f"{cls.__name__}.from_callable() expects a plain callable; use {cls.__name__}.ensure_factory()."
            )
        return cls(fn, from_config=_wrap_as_factory(fn), signature_source=fn)

    @classmethod
    def ensure_factory(
        cls: type[FactoryT],
        fn: Callable[..., FactoryOutputT] | Factory[Any],
    ) -> FactoryT:
        """Normalize a plain callable or existing compatible factory."""
        if isinstance(fn, cls):
            return fn
        if isinstance(fn, Factory):
            if type(fn) is Factory:
                return cls(fn._fn, from_config=fn._from_config, signature_source=fn._signature_source)
            raise TypeError(
                f"{cls.__name__}.ensure_factory() expects a callable or {cls.__name__}, "
                f"got {type(fn).__name__}."
            )
        return cls.from_callable(fn)

    def from_config(self, config: dict) -> FactoryOutputT:
        self.validate_config(config, name=self._factory_name)
        return self._validate_output(self._from_config(config), config=config)

    def validate_config(self, config: dict, *, name: str = "factory") -> None:
        """Validate a config dict against the wrapped keyword signature."""
        target = self._signature_source
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

    def __call__(self, *args, **kwargs) -> FactoryOutputT:
        return self._validate_output(self._fn(*args, **kwargs))

    def _validate_output(
        self,
        output: FactoryOutputT,
        *,
        config: dict | None = None,
    ) -> FactoryOutputT:
        return output

    @classmethod
    def _discover_factory(
        cls: type[FactoryT],
        module: Any,
        explicit_fn: Any,
    ) -> FactoryT | None:
        if explicit_fn is not None:
            return cls.ensure_factory(explicit_fn)
        return _discover_factory_in_module(module, cls)


class PipelineFactory(Factory[Pipeline]):
    """Factory subtype for pipeline-producing callables."""

    _factory_name = "pipeline factory"

    @classmethod
    def discover(cls, module: Any, explicit_fn: Any = None) -> PipelineFactory | None:
        """Discover the module's pipeline factory, normalized to ``PipelineFactory``."""
        return cls._discover_factory(module, explicit_fn)

    def _validate_output(
        self,
        output: Any,
        *,
        config: dict | None = None,
    ) -> Pipeline:
        if isinstance(output, Pipeline):
            return output

        config_suffix = f" for config {config!r}" if config is not None else ""
        raise TypeError(
            f"{self._factory_name} must return a Pipeline, got {type(output).__name__!r}{config_suffix}"
        )


class DataFactory(Factory[InputFn]):
    """Factory subtype for input-producing callables."""

    _factory_name = "data factory"

    @classmethod
    def discover(cls, module: Any, explicit_fn: Any = None) -> DataFactory | None:
        """Discover the module's data factory, normalized to ``DataFactory``."""
        return cls._discover_factory(module, explicit_fn)

    def _validate_output(
        self,
        output: Any,
        *,
        config: dict | None = None,
    ) -> InputFn:
        if callable(output):
            return output

        config_suffix = f" for config {config!r}" if config is not None else ""
        raise TypeError(
            f"{self._factory_name} must return a callable InputFn, "
            f"got {type(output).__name__!r}{config_suffix}"
        )


def pipeline_factory(fn: Callable[..., Pipeline]) -> PipelineFactory:
    """Wrap a function as a discoverable pipeline factory.

    The decorated value is a ``PipelineFactory``. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. Any parameter without a default must be supplied
    through ``--arg``, ``--config``, or ``--axis``.
    """
    return PipelineFactory.from_callable(fn)


def data_factory(fn: Callable[..., InputFn]) -> DataFactory:
    """Wrap a function as a discoverable data factory.

    The decorated value is a ``DataFactory``. It preserves the original
    Python call semantics while also exposing ``from_config(config)`` for CLI
    and benchmark helpers. It must return an ``InputFn`` — a zero-argument
    callable yielding ``(id: str, value: Any, tag: str | None,
    metadata: dict | None)``.
    """
    return DataFactory.from_callable(fn)


def _wrap_as_factory(
    fn: Callable[..., FactoryOutputT],
) -> Callable[[dict], FactoryOutputT]:
    """Wrap a callable with a ``from_config(config)`` adapter."""
    @functools.wraps(fn)
    def wrapper(config: dict) -> FactoryOutputT:
        return fn(**config)

    return wrapper


def _discover_factory_in_module(
    module: Any,
    factory_type: type[FactoryT],
) -> FactoryT | None:
    kind = factory_type.__name__.removesuffix("Factory").lower()
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
