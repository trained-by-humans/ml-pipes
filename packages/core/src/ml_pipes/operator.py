from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

_CAPTURED_ARGUMENTS_ATTR = "__ml_pipes_operator_argument_entries__"
_CONSTRUCTOR_SIGNATURE_ATTR = "__ml_pipes_operator_constructor_signature__"
_EXCLUDED_OPERATOR_ARGUMENTS_ATTR = "__ml_pipes_operator_excluded_arguments__"
_WRAPPED_ATTR = "__ml_pipes_operator_wrapped__"
_EXCLUDED_ARGUMENTS = {"self"}
_CONCISE_COLLECTION_ITEM_LIMIT = 3

_OperatorType = TypeVar("_OperatorType", bound=type)


@dataclass(frozen=True)
class OperatorArgument:
    name: str
    value: Any
    kind: inspect._ParameterKind
    source: Literal["passed", "default"] = "passed"
    passed_by_keyword: bool = False

    @property
    def is_default(self) -> bool:
        return self.source == "default"

    @property
    def is_passed(self) -> bool:
        return self.source == "passed"

    @staticmethod
    def to_dict(arguments: Any) -> dict[str, Any]:
        return {
            argument.name: argument.value
            for argument in arguments
        }


@dataclass(frozen=True)
class OperatorDescription:
    name: str
    arguments: tuple[OperatorArgument, ...] = field(default_factory=tuple)
    constructor_signature: inspect.Signature | None = None
    bare_callable: bool = False

    @classmethod
    def from_operator(cls, operator: Any) -> "OperatorDescription":
        return cls(
            name=get_operator_name(operator),
            arguments=get_operator_argument_entries(operator),
            constructor_signature=get_operator_constructor_signature(operator),
            bare_callable=_is_bare_callable(operator),
        )

    @property
    def passed_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(argument for argument in self.arguments if argument.is_passed)

    @property
    def default_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(argument for argument in self.arguments if argument.is_default)

    @property
    def all_args(self) -> dict[str, Any]:
        return OperatorArgument.to_dict(self.arguments)

    def render(
        self,
        *,
        show_defaults: bool = False,
        verbose: bool = False,
    ) -> str:
        return _format_operator_description(
            name=self.name,
            arguments=self.arguments,
            bare_callable=self.bare_callable,
            show_defaults=show_defaults,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__


def Operator(cls: _OperatorType) -> _OperatorType:
    """Capture constructor arguments for operator instances."""
    if not inspect.isclass(cls):
        raise TypeError("@Operator can only be applied to classes")

    if not _supports_instance_attribute(cls, _CAPTURED_ARGUMENTS_ATTR):
        raise TypeError(
            "@Operator requires classes that can store captured constructor "
            f"args; {cls.__name__} does not allow the internal "
            f"{_CAPTURED_ARGUMENTS_ATTR!r} attribute"
        )
    init = cls.__init__
    target = getattr(init, "__wrapped__", init)
    constructor_signature = inspect.signature(target)
    excluded_arguments = _excluded_argument_names(cls)
    public_constructor_signature = _public_constructor_signature(
        constructor_signature,
        excluded_arguments,
    )
    setattr(cls, _CONSTRUCTOR_SIGNATURE_ATTR, public_constructor_signature)

    if _effective_repr(cls) is object.__repr__:
        def _default_operator_repr(self: Any) -> str:
            try:
                description = self.describe()
            except Exception:
                description = None

            if isinstance(description, OperatorDescription):
                return description.render()
            return OperatorDescription.from_operator(self).render()

        cls.__repr__ = _default_operator_repr

    if getattr(cls, "describe", None) is None:
        def _default_operator_describe(self: Any) -> OperatorDescription:
            return OperatorDescription.from_operator(self)

        cls.describe = _default_operator_describe

    if getattr(init, _WRAPPED_ATTR, False):
        return cls

    @functools.wraps(target)
    def wrapped(self, *args, **kwargs):
        constructor_signature.bind(self, *args, **kwargs)
        captured_arguments = _capture_arguments(
            constructor_signature,
            args,
            kwargs,
            excluded_arguments=excluded_arguments,
        )
        object.__setattr__(self, _CAPTURED_ARGUMENTS_ATTR, captured_arguments)
        return target(self, *args, **kwargs)

    wrapped.__signature__ = constructor_signature
    setattr(wrapped, _WRAPPED_ATTR, True)
    cls.__init__ = wrapped
    return cls


def get_operator_name(operator: Any) -> str:
    if inspect.isfunction(operator) or inspect.ismethod(operator) or inspect.isbuiltin(operator):
        return getattr(operator, "__name__", type(operator).__name__)
    if inspect.isclass(operator):
        return operator.__name__
    return type(operator).__name__


def get_operator_constructor_signature(operator: Any) -> inspect.Signature | None:
    if inspect.isfunction(operator) or inspect.ismethod(operator) or inspect.isbuiltin(operator):
        return None

    target = operator if inspect.isclass(operator) else type(operator)
    constructor_signature = getattr(target, _CONSTRUCTOR_SIGNATURE_ATTR, None)
    if isinstance(constructor_signature, inspect.Signature):
        return constructor_signature

    init = getattr(target, "__init__", None)
    if init is None:
        return None

    raw_target = getattr(init, "__wrapped__", init)
    try:
        return _public_constructor_signature(
            inspect.signature(raw_target),
            _excluded_argument_names(target),
        )
    except (TypeError, ValueError):
        return None


def get_operator_argument_entries(
    operator: Any,
) -> tuple[OperatorArgument, ...]:
    captured = getattr(operator, _CAPTURED_ARGUMENTS_ATTR, None)
    if isinstance(captured, tuple) and all(isinstance(argument, OperatorArgument) for argument in captured):
        return captured

    return ()


def _capture_arguments(
    constructor_signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    excluded_arguments: set[str],
) -> tuple[OperatorArgument, ...]:
    captured: list[OperatorArgument] = []
    consumed_keywords: set[str] = set()
    positional_index = 0
    parameters = tuple(constructor_signature.parameters.values())

    for parameter in parameters:
        if parameter.name in excluded_arguments:
            continue

        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            if positional_index < len(args):
                captured.append(
                    OperatorArgument(
                        name=parameter.name,
                        value=tuple(args[positional_index:]),
                        kind=parameter.kind,
                        passed_by_keyword=False,
                    )
                )
                positional_index = len(args)
            continue

        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extra_kwargs = {
                name: value
                for name, value in kwargs.items()
                if name not in consumed_keywords
            }
            if extra_kwargs:
                captured.append(
                    OperatorArgument(
                        name=parameter.name,
                        value=extra_kwargs,
                        kind=parameter.kind,
                        passed_by_keyword=True,
                    )
                )
            continue

        if positional_index < len(args):
            captured.append(
                OperatorArgument(
                    name=parameter.name,
                    value=args[positional_index],
                    kind=parameter.kind,
                    passed_by_keyword=False,
                )
            )
            positional_index += 1
            continue

        if parameter.name not in kwargs:
            if parameter.default is inspect.Parameter.empty:
                continue

            captured.append(
                OperatorArgument(
                    name=parameter.name,
                    value=parameter.default,
                    kind=parameter.kind,
                    source="default",
                    passed_by_keyword=parameter.kind is inspect.Parameter.KEYWORD_ONLY,
                )
            )
            continue

        consumed_keywords.add(parameter.name)
        captured.append(
            OperatorArgument(
                name=parameter.name,
                value=kwargs[parameter.name],
                kind=parameter.kind,
                passed_by_keyword=True,
            )
        )

    return tuple(captured)


def _excluded_argument_names(target: Any) -> set[str]:
    cls = target if inspect.isclass(target) else type(target)
    excluded = getattr(cls, _EXCLUDED_OPERATOR_ARGUMENTS_ATTR, ())
    if isinstance(excluded, str):
        excluded = (excluded,)
    return _EXCLUDED_ARGUMENTS.union(excluded)


def _public_constructor_signature(
    constructor_signature: inspect.Signature,
    excluded_arguments: set[str],
) -> inspect.Signature:
    return constructor_signature.replace(
        parameters=[
            parameter
            for parameter in constructor_signature.parameters.values()
            if parameter.name not in excluded_arguments
        ]
    )


def _supports_instance_attribute(cls: type, name: str) -> bool:
    for current in cls.__mro__[:-1]:
        slots = current.__dict__.get("__slots__", None)
        if slots is None:
            return True

        if isinstance(slots, str):
            slots = (slots,)

        if name in slots or "__dict__" in slots:
            return True

    return False


def _format_operator_description(
    *,
    name: str,
    arguments: tuple[OperatorArgument, ...],
    bare_callable: bool,
    show_defaults: bool,
    verbose: bool,
) -> str:
    if bare_callable:
        return name

    rendered_arguments = _format_call_arguments(
        _select_render_arguments(
            arguments,
            include_defaults=show_defaults,
            verbose=verbose,
        ),
        verbose=verbose,
    )
    return f"{name}({rendered_arguments})"


def _select_render_arguments(
    arguments: tuple[OperatorArgument, ...],
    *,
    include_defaults: bool,
    verbose: bool,
) -> tuple[OperatorArgument, ...]:
    visible_arguments = arguments if include_defaults else tuple(
        argument for argument in arguments if argument.is_passed
    )
    if verbose:
        return visible_arguments
    return tuple(
        argument for argument in visible_arguments
        if not _is_pipeline_instance(argument.value)
    )


def _format_call_arguments(
    arguments: tuple[OperatorArgument, ...],
    *,
    verbose: bool,
) -> str:
    parts: list[str] = []
    for argument in arguments:
        if argument.kind is inspect.Parameter.POSITIONAL_ONLY:
            parts.append(_format_arg_value(argument.value, verbose=verbose))
            continue
        if argument.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.extend(_format_arg_value(item, verbose=verbose) for item in argument.value)
            continue
        if argument.kind is inspect.Parameter.VAR_KEYWORD:
            parts.extend(
                f"{name}={_format_arg_value(item, verbose=verbose)}"
                for name, item in argument.value.items()
            )
            continue
        if argument.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and not argument.passed_by_keyword and argument.is_passed:
            parts.append(_format_arg_value(argument.value, verbose=verbose))
            continue
        parts.append(f"{argument.name}={_format_arg_value(argument.value, verbose=verbose)}")
    return ", ".join(parts)


def _format_arg_value(value: Any, *, verbose: bool) -> str:
    if _is_pipeline_instance(value):
        return repr(value) if verbose else type(value).__name__
    if isinstance(value, tuple):
        return _format_tuple_value(value, verbose=verbose)
    if isinstance(value, list):
        return _format_list_value(value, verbose=verbose)
    if isinstance(value, dict):
        return _format_dict_value(value, verbose=verbose)
    if isinstance(value, set):
        return _format_set_value(value, verbose=verbose)
    if isinstance(value, frozenset):
        return _format_frozenset_value(value, verbose=verbose)
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return repr(value)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return _format_callable_value(value)
    if inspect.isclass(value):
        return value.__name__
    if callable(value):
        if _effective_repr(type(value)) is object.__repr__:
            return type(value).__name__
        return repr(value)
    return repr(value)


def _format_tuple_value(value: tuple[Any, ...], *, verbose: bool) -> str:
    items, truncated = _truncate_collection_items(value, verbose=verbose)
    rendered_items = [_format_arg_value(item, verbose=verbose) for item in items]

    if len(value) == 1:
        return f"({rendered_items[0]},)"

    if truncated:
        rendered_items.append("...")
    return "(" + ", ".join(rendered_items) + ")"


def _format_list_value(value: list[Any], *, verbose: bool) -> str:
    items, truncated = _truncate_collection_items(value, verbose=verbose)
    rendered_items = [_format_arg_value(item, verbose=verbose) for item in items]
    if truncated:
        rendered_items.append("...")
    return "[" + ", ".join(rendered_items) + "]"


def _format_dict_value(value: dict[Any, Any], *, verbose: bool) -> str:
    items, truncated = _truncate_collection_items(tuple(value.items()), verbose=verbose)
    rendered_items = [
        f"{_format_arg_value(key, verbose=verbose)}: {_format_arg_value(item, verbose=verbose)}"
        for key, item in items
    ]
    if truncated:
        rendered_items.append("...")
    return "{" + ", ".join(rendered_items) + "}"


def _format_set_value(value: set[Any], *, verbose: bool) -> str:
    if not value:
        return "set()"
    return "{" + ", ".join(_format_set_items(value, verbose=verbose)) + "}"


def _format_frozenset_value(value: frozenset[Any], *, verbose: bool) -> str:
    if not value:
        return "frozenset()"
    return "frozenset({" + ", ".join(_format_set_items(value, verbose=verbose)) + "})"


def _format_set_items(value: set[Any] | frozenset[Any], *, verbose: bool) -> list[str]:
    rendered_items = sorted(_format_arg_value(item, verbose=verbose) for item in value)
    if not verbose and len(rendered_items) > _CONCISE_COLLECTION_ITEM_LIMIT:
        return rendered_items[:_CONCISE_COLLECTION_ITEM_LIMIT] + ["..."]
    return rendered_items


def _truncate_collection_items(
    items: tuple[Any, ...] | list[Any],
    *,
    verbose: bool,
) -> tuple[tuple[Any, ...] | list[Any], bool]:
    if verbose or len(items) <= _CONCISE_COLLECTION_ITEM_LIMIT:
        return items, False
    return items[:_CONCISE_COLLECTION_ITEM_LIMIT], True


def _format_callable_value(value: Any) -> str:
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return getattr(value, "__name__", type(value).__name__)
    if inspect.isclass(value):
        return value.__name__
    if _effective_repr(type(value)) is object.__repr__:
        return type(value).__name__
    return repr(value)


def _is_pipeline_instance(value: Any) -> bool:
    try:
        from .core import Pipeline
    except Exception:
        return False
    return isinstance(value, Pipeline)


def _is_bare_callable(value: Any) -> bool:
    return (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or inspect.isbuiltin(value)
        or inspect.isclass(value)
    )


def _effective_repr(target: type) -> Any:
    return getattr(target, "__repr__", object.__repr__)
