from __future__ import annotations

import functools
import inspect
from typing import Any, TypeVar

_CAPTURED_ARGS_ATTR = "__ml_pipes_operator_args__"
_SIGNATURE_ATTR = "__ml_pipes_operator_signature__"
_WRAPPED_ATTR = "__ml_pipes_operator_wrapped__"
_EXCLUDED_ARGUMENTS = {"self", "pipeline"}

_OperatorType = TypeVar("_OperatorType", bound=type)


def Operator(cls: _OperatorType) -> _OperatorType:
    """Capture constructor arguments for operator instances."""
    if not inspect.isclass(cls):
        raise TypeError("@Operator can only be applied to classes")

    init = cls.__init__
    target = getattr(init, "__wrapped__", init)
    signature = inspect.signature(target)
    public_signature = _public_signature(signature)
    setattr(cls, _SIGNATURE_ATTR, public_signature)

    if getattr(init, _WRAPPED_ATTR, False):
        return cls

    @functools.wraps(target)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        object.__setattr__(self, _CAPTURED_ARGS_ATTR, _capture_arguments(signature, bound.arguments))
        return target(self, *args, **kwargs)

    wrapped.__signature__ = signature
    setattr(wrapped, _WRAPPED_ATTR, True)
    cls.__init__ = wrapped
    return cls


def get_operator_signature(operator: Any) -> inspect.Signature | None:
    target = operator if inspect.isclass(operator) else type(operator)
    signature = getattr(target, _SIGNATURE_ATTR, None)
    if isinstance(signature, inspect.Signature):
        return signature

    init = getattr(target, "__init__", None)
    if init is None:
        return None

    raw_target = getattr(init, "__wrapped__", init)
    try:
        return _public_signature(inspect.signature(raw_target))
    except (TypeError, ValueError):
        return None


def get_operator_args(operator: Any, *, include_defaults: bool = False) -> dict[str, Any]:
    captured = getattr(operator, _CAPTURED_ARGS_ATTR, None)
    if not isinstance(captured, dict):
        return {}

    result = dict(captured)
    if not include_defaults:
        return result

    signature = get_operator_signature(operator)
    if signature is None:
        return result

    for name, parameter in signature.parameters.items():
        if name in result:
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        result[name] = parameter.default

    return result


def _capture_arguments(
    signature: inspect.Signature,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in _EXCLUDED_ARGUMENTS:
            continue
        if name not in arguments:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            captured[name] = tuple(arguments[name])
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            captured[name] = dict(arguments[name])
            continue
        captured[name] = arguments[name]
    return captured


def _public_signature(signature: inspect.Signature) -> inspect.Signature:
    return signature.replace(
        parameters=[
            parameter
            for parameter in signature.parameters.values()
            if parameter.name not in _EXCLUDED_ARGUMENTS
        ]
    )
