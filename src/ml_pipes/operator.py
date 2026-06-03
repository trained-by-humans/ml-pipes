from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

_CAPTURED_ARGUMENTS_ATTR = "__ml_pipes_operator_argument_entries__"
_CONSTRUCTOR_SIGNATURE_ATTR = "__ml_pipes_operator_constructor_signature__"
_WRAPPED_ATTR = "__ml_pipes_operator_wrapped__"
_EXCLUDED_ARGUMENTS = {"self", "pipeline"}

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


def Operator(cls: _OperatorType) -> _OperatorType:
    """Capture constructor arguments for operator instances."""
    if not inspect.isclass(cls):
        raise TypeError("@Operator can only be applied to classes")

    init = cls.__init__
    target = getattr(init, "__wrapped__", init)
    constructor_signature = inspect.signature(target)
    public_constructor_signature = _public_constructor_signature(constructor_signature)
    setattr(cls, _CONSTRUCTOR_SIGNATURE_ATTR, public_constructor_signature)

    if _effective_repr(cls) is object.__repr__:
        cls.__repr__ = _describe_captured_operator

    if getattr(cls, "describe", None) is None:
        cls.describe = _describe_captured_operator

    if getattr(init, _WRAPPED_ATTR, False):
        return cls

    @functools.wraps(target)
    def wrapped(self, *args, **kwargs):
        constructor_signature.bind(self, *args, **kwargs)
        captured_arguments = _capture_arguments(constructor_signature, args, kwargs)
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
        return _public_constructor_signature(inspect.signature(raw_target))
    except (TypeError, ValueError):
        return None


def get_operator_argument_entries(
    operator: Any,
) -> tuple[OperatorArgument, ...]:
    captured = getattr(operator, _CAPTURED_ARGUMENTS_ATTR, None)
    if isinstance(captured, tuple) and all(isinstance(argument, OperatorArgument) for argument in captured):
        return captured

    return ()


def render_operator(
    operator: Any,
    *,
    show_defaults: bool = False,
    mode: Literal["repr", "describe"] = "repr",
    name: str | None = None,
    arguments: tuple[OperatorArgument, ...] | None = None,
) -> str:
    operator_name = name or get_operator_name(operator)
    operator_arguments = arguments
    if operator_arguments is None:
        operator_arguments = get_operator_argument_entries(operator)
    operator_arguments = _filter_arguments(operator_arguments, include_defaults=show_defaults)
    return _render_operator_mode(
        operator,
        name=operator_name,
        arguments=operator_arguments,
        show_defaults=show_defaults,
        mode=mode,
    )


def _capture_arguments(
    constructor_signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[OperatorArgument, ...]:
    captured: list[OperatorArgument] = []
    consumed_keywords: set[str] = set()
    positional_index = 0
    parameters = tuple(constructor_signature.parameters.values())

    for parameter in parameters:
        if parameter.name in _EXCLUDED_ARGUMENTS:
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

def _describe_captured_operator(
    operator: Any,
    *,
    show_defaults: bool = False,
) -> str:
    return _format_operator_call(
        operator=operator,
        name=get_operator_name(operator),
        arguments=_filter_arguments(
            get_operator_argument_entries(operator),
            include_defaults=show_defaults,
        ),
    )


def _render_operator_mode(
    operator: Any,
    *,
    name: str,
    arguments: tuple[OperatorArgument, ...],
    show_defaults: bool,
    mode: Literal["repr", "describe"],
) -> str:
    if mode == "describe":
        description = _try_operator_describe(operator, show_defaults=show_defaults)
        if description is not None:
            return description

        if not show_defaults:
            representation = _try_operator_repr(operator)
            if representation is not None:
                return representation

        return _format_operator_call(operator=operator, name=name, arguments=arguments)

    if mode == "repr":
        representation = _try_operator_repr(operator)
        if representation is not None:
            return representation
        return _format_operator_call(operator=operator, name=name, arguments=arguments)

    raise ValueError(f"Unsupported operator render mode: {mode!r}")


def _try_operator_describe(operator: Any, *, show_defaults: bool) -> str | None:
    method = getattr(operator, "describe", None)
    if not callable(method):
        return None

    try:
        description = _invoke_operator_describe(method, show_defaults=show_defaults)
    except Exception:
        return None

    if isinstance(description, str):
        return description
    return None


def _invoke_operator_describe(method: Any, *, show_defaults: bool) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(show_defaults=show_defaults)

    parameters = tuple(signature.parameters.values())
    if (
        "show_defaults" in signature.parameters
        or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    ):
        return method(show_defaults=show_defaults)
    if not parameters:
        return method()
    return method()


def _try_operator_repr(operator: Any) -> str | None:
    if _is_bare_callable(operator):
        return None
    if _effective_repr(type(operator)) is object.__repr__:
        return None

    try:
        return repr(operator)
    except Exception:
        return None


def _format_operator_call(
    *,
    operator: Any,
    name: str,
    arguments: tuple[OperatorArgument, ...],
) -> str:
    if _is_bare_callable(operator):
        return name

    return f"{name}({_format_call_arguments(arguments)})"


def _format_call_arguments(arguments: tuple[OperatorArgument, ...]) -> str:
    parts: list[str] = []
    for argument in arguments:
        if argument.kind is inspect.Parameter.POSITIONAL_ONLY:
            parts.append(_format_arg_value(argument.value))
            continue
        if argument.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.extend(_format_arg_value(item) for item in argument.value)
            continue
        if argument.kind is inspect.Parameter.VAR_KEYWORD:
            parts.extend(
                f"{name}={_format_arg_value(item)}"
                for name, item in argument.value.items()
            )
            continue
        if argument.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and not argument.passed_by_keyword and argument.is_passed:
            parts.append(_format_arg_value(argument.value))
            continue
        parts.append(f"{argument.name}={_format_arg_value(argument.value)}")
    return ", ".join(parts)


def _format_arg_value(value: Any) -> str:
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({_format_arg_value(value[0])},)"
        return "(" + ", ".join(_format_arg_value(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_format_arg_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_format_arg_value(key)}: {_format_arg_value(item)}"
            for key, item in value.items()
        ) + "}"
    if isinstance(value, set):
        if not value:
            return "set()"
        items = sorted(_format_arg_value(item) for item in value)
        return "{" + ", ".join(items) + "}"
    if isinstance(value, frozenset):
        if not value:
            return "frozenset()"
        items = sorted(_format_arg_value(item) for item in value)
        return "frozenset({" + ", ".join(items) + "})"
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return repr(value)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return _callable_label(value)
    if inspect.isclass(value):
        return value.__name__
    if callable(value):
        if _effective_repr(type(value)) is object.__repr__:
            return type(value).__name__
        return repr(value)
    return repr(value)


def _callable_label(value: Any) -> str:
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return getattr(value, "__name__", type(value).__name__)
    if inspect.isclass(value):
        return value.__name__
    if _effective_repr(type(value)) is object.__repr__:
        return type(value).__name__
    return repr(value)


def _filter_arguments(
    arguments: tuple[OperatorArgument, ...],
    *,
    include_defaults: bool,
) -> tuple[OperatorArgument, ...]:
    if include_defaults:
        return arguments
    return tuple(argument for argument in arguments if argument.is_passed)


def _is_bare_callable(value: Any) -> bool:
    return (
        inspect.isfunction(value)
        or inspect.ismethod(value)
        or inspect.isbuiltin(value)
        or inspect.isclass(value)
    )


def _effective_repr(target: type) -> Any:
    return getattr(target, "__repr__", object.__repr__)


def _public_constructor_signature(constructor_signature: inspect.Signature) -> inspect.Signature:
    return constructor_signature.replace(
        parameters=[
            parameter
            for parameter in constructor_signature.parameters.values()
            if parameter.name not in _EXCLUDED_ARGUMENTS
        ]
    )
