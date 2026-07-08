from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Generic, TypeVar, cast

from ml_pipes.pipeline import Pipeline

FactoryOutputT = TypeVar("FactoryOutputT")
FactoryT = TypeVar("FactoryT", bound="Factory[Any]")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


class Factory(Generic[FactoryOutputT]):
    """Callable wrapper with a config-driven entrypoint."""

    _factory_name = "factory"

    def __init__(self, source: Callable[..., FactoryOutputT]) -> None:
        self._source = source
        functools.update_wrapper(self, source)

    def __call__(self, *args, **kwargs) -> FactoryOutputT:
        return self._validate_output(self._source(*args, **kwargs))

    def build(self, config: dict) -> FactoryOutputT:
        self.validate_config(config, name=self._factory_name)
        output = self._source(**config)
        return self._validate_output(output, config=config)

    def validate_config(self, config: dict, *, name: str = "factory") -> None:
        """Validate a config dict against the wrapped keyword signature."""
        params = inspect.signature(self._source).parameters
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

    def _validate_output(
        self,
        output: FactoryOutputT,
        *,
        config: dict | None = None,
    ) -> FactoryOutputT:
        return output

    @classmethod
    def from_callable(
        cls: type[FactoryT],
        source: Callable[..., FactoryOutputT],
    ) -> FactoryT:
        """Build a ``Factory`` from a callable invoked as ``source(**config)``."""
        if isinstance(source, Factory):
            raise TypeError(
                f"{cls.__name__}.from_callable() expects a plain callable; use {cls.__name__}.ensure_factory()."
            )
        annotated = _find_annotated_factory(source)
        if annotated is not None:
            decorator = _factory_decorator_name(type(annotated))
            raise TypeError(
                f"{cls.__name__}.from_callable() cannot wrap a callable that already wraps {decorator}. "
                f"{decorator} must be the outermost decorator."
            )
        return cls(source)

    @classmethod
    def ensure_factory(
        cls: type[FactoryT],
        source: Callable[..., FactoryOutputT] | Factory[FactoryOutputT],
    ) -> FactoryT:
        """Normalize a plain callable or existing compatible factory."""
        if isinstance(source, cls):
            return source
        if isinstance(source, Factory):
            if type(source) is Factory:
                return cls(source._source)
            raise TypeError(
                f"{cls.__name__}.ensure_factory() expects a callable or {cls.__name__}, "
                f"got {type(source).__name__}."
            )
        return cls.from_callable(source)

    @classmethod
    def discover(
        cls: type[FactoryT],
        module: Any,
        explicit_fn: Any = None,
    ) -> FactoryT | None:
        """Discover the module's factory exported as ``cls``."""
        if explicit_fn is not None:
            return cls.ensure_factory(explicit_fn)
        return _discover_factory_in_module(module, cls)


class PipelineFactory(Factory[Pipeline[InputT, OutputT]], Generic[InputT, OutputT]):
    """Factory subtype for pipeline-producing callables."""

    _factory_name = "pipeline factory"

    @classmethod
    def from_callable(
        cls,
        source: Callable[..., Pipeline[InputT, OutputT]],
    ) -> "PipelineFactory[InputT, OutputT]":
        return cast("PipelineFactory[InputT, OutputT]", super().from_callable(source))

    @classmethod
    def ensure_factory(
        cls,
        source: (
            Callable[..., Pipeline[InputT, OutputT]]
            | Factory[Pipeline[InputT, OutputT]]
        ),
    ) -> "PipelineFactory[InputT, OutputT]":
        return cast("PipelineFactory[InputT, OutputT]", super().ensure_factory(source))

    def _validate_output(
        self,
        output: Any,
        *,
        config: dict | None = None,
    ) -> Pipeline[InputT, OutputT]:
        if isinstance(output, Pipeline):
            return cast(Pipeline[InputT, OutputT], output)

        config_suffix = f" for config {config!r}" if config is not None else ""
        raise TypeError(
            f"{self._factory_name} must return a Pipeline, got {type(output).__name__!r}{config_suffix}"
        )


class DataFactory(Factory[InputFn]):
    """Factory subtype for input-producing callables."""

    _factory_name = "data factory"

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


def pipeline_factory(
    fn: Callable[..., Pipeline[InputT, OutputT]],
) -> PipelineFactory[InputT, OutputT]:
    """Wrap a declared reusable function as a discoverable pipeline factory.

    Place ``@pipeline_factory`` on the top decorator line. If another
    decorator wraps the resulting factory object, discovery and factory
    normalization raise an error instead of adapting it.

    The decorated value stays directly callable and also exposes
    ``build(config)`` for CLI and benchmark helpers. Any parameter
    without a default must be supplied through ``--arg``, ``--config``,
    or ``--axis``.
    """
    return PipelineFactory.from_callable(fn)


def data_factory(fn: Callable[..., InputFn]) -> DataFactory:
    """Wrap a declared reusable function as a discoverable data factory.

    Place ``@data_factory`` on the top decorator line. If another decorator
    wraps the resulting factory object, discovery and factory normalization
    raise an error instead of adapting it.

    The decorated value stays directly callable and also exposes
    ``build(config)`` for CLI and benchmark helpers. It must return an
    ``InputFn`` — a zero-argument callable yielding ``(id: str, value: Any,
    tag: str | None, metadata: dict | None)``.
    """
    return DataFactory.from_callable(fn)


def _discover_factory_in_module(
    module: Any,
    factory_type: type[FactoryT],
) -> FactoryT | None:
    kind = factory_type.__name__.removesuffix("Factory").lower()
    found: list[tuple[str, FactoryT]] = []
    for name, source in vars(module).items():
        if isinstance(source, factory_type):
            found.append((name, source))
            continue

        annotated = _find_annotated_factory(source)
        if annotated is not None and isinstance(annotated, factory_type):
            decorator = _factory_decorator_name(factory_type)
            raise TypeError(
                f"{module.__name__!r}.{name} wraps {decorator}. "
                f"{decorator} must be the outermost decorator."
            )

    if len(found) > 1:
        names = ", ".join(name for name, _ in found)
        raise ValueError(
            f"multiple @{kind}_factory found in {module.__name__!r}: [{names}]. "
            f"Only one is allowed per module; remove one or use 'module:{kind}_factory_fn' syntax."
        )

    return found[0][1] if found else None


def _find_annotated_factory(source: Any) -> Factory[Any] | None:
    current = source
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Factory):
            return current
        current = getattr(current, "__wrapped__", None)
    return None


def _factory_decorator_name(factory_type: type[Factory[Any]]) -> str:
    kind = factory_type.__name__.removesuffix("Factory").lower()
    return f"@{kind}_factory"
