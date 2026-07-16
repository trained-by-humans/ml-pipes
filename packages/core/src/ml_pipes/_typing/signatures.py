from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable
from typing import get_type_hints


_POSITIONAL_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
_POSITIONAL_VALUE_PARAMETER_KINDS = _POSITIONAL_PARAMETER_KINDS + (
    inspect.Parameter.VAR_POSITIONAL,
)


@dataclass(frozen=True)
class CallableParameterAnnotation:
    parameter: inspect.Parameter
    annotation: Any | None


@dataclass(frozen=True)
class CallableSignatureAnnotations:
    parameters: tuple[CallableParameterAnnotation, ...]
    return_annotation: Any | None
    is_inspectable: bool

    @property
    def parameter_annotations(self) -> tuple[Any | None, ...]:
        return tuple(parameter.annotation for parameter in self.parameters)


def resolve_callable_signature_annotations(
    callable_: Callable[..., Any],
) -> CallableSignatureAnnotations:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return CallableSignatureAnnotations((), None, False)

    parameters = tuple(signature.parameters.values())
    try:
        hints = _resolve_callable_hints(callable_)
    except (TypeError, ValueError):
        return CallableSignatureAnnotations(
            tuple(
                CallableParameterAnnotation(parameter, None)
                for parameter in parameters
            ),
            None,
            True,
        )

    return_annotation = callable_ if inspect.isclass(callable_) else hints.get("return")
    return CallableSignatureAnnotations(
        tuple(
            CallableParameterAnnotation(parameter, hints.get(parameter.name))
            for parameter in parameters
        ),
        return_annotation,
        True,
    )


def match_method_signatures(
    source_owner: type,
    target_owner: type,
    member: str,
) -> tuple[CallableSignatureAnnotations, CallableSignatureAnnotations] | None:
    source_signature = _resolve_method_signature_annotations(
        source_owner,
        member,
    )
    target_signature = _resolve_method_signature_annotations(
        target_owner,
        member,
    )
    if source_signature is None or target_signature is None:
        return None
    source_signature = _strip_member_receiver(
        source_owner,
        member,
        source_signature,
    )
    target_signature = _strip_member_receiver(
        target_owner,
        member,
        target_signature,
    )
    if source_signature is None or target_signature is None:
        return None
    if (
        source_signature.return_annotation is None
        or target_signature.return_annotation is None
    ):
        return None
    if any(
        parameter.annotation is None
        or parameter.parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in (
            *source_signature.parameters,
            *target_signature.parameters,
        )
    ):
        return None
    if not _parameter_signatures_match(
        tuple(parameter.parameter for parameter in source_signature.parameters),
        tuple(parameter.parameter for parameter in target_signature.parameters),
    ):
        return None

    return source_signature, target_signature


def _parameter_signatures_match(
    source_parameters: tuple[inspect.Parameter, ...],
    target_parameters: tuple[inspect.Parameter, ...],
) -> bool:
    if len(source_parameters) != len(target_parameters):
        return False

    for source_parameter, target_parameter in zip(
        source_parameters,
        target_parameters,
    ):
        if source_parameter.kind is not target_parameter.kind:
            return False
        if source_parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            return False
        if (
            source_parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
            and source_parameter.name != target_parameter.name
        ):
            return False
        if not _parameter_defaults_match(
            source_parameter.default,
            target_parameter.default,
        ):
            return False

    return True


def validate_operator_signature(
    operator: Any,
    *,
    label: str,
    error_type: type[Exception],
    warning_type: type[Warning] | None = None,
) -> tuple[inspect.Parameter, ...]:
    try:
        signature = inspect.signature(operator)
    except (TypeError, ValueError) as exc:
        if not callable(operator):
            raise error_type(
                f"Pipeline step {label} must define __call__"
            ) from exc
        raise error_type(
            f"Pipeline step {label} must expose an inspectable call signature"
        ) from exc
    parameters = tuple(signature.parameters.values())
    variadic_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL
    )
    if variadic_parameters:
        raise error_type(
            f"Pipeline step {label} uses variadic positional parameters "
            f"({_format_parameter_descriptions(variadic_parameters)}), "
            f"which Pipeline does not support. Use a single tuple-typed parameter instead."
        )

    unsupported_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind not in _POSITIONAL_PARAMETER_KINDS
    )
    if unsupported_parameters:
        raise error_type(
            f"Pipeline step {label} uses non-positional parameters "
            f"({_format_parameter_descriptions(unsupported_parameters)}), but Pipeline chains operators by "
            f"argument position. Use only positional parameters in __call__."
        )

    if not parameters:
        raise error_type(
            f"Pipeline step {label} must define at least one positional input parameter in __call__"
        )

    if warning_type is not None and len(parameters) > 1:
        defaulted_parameters = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in _POSITIONAL_PARAMETER_KINDS
            and parameter.default is not inspect.Parameter.empty
        )
        if defaulted_parameters:
            defaulted_parameter_names = ", ".join(
                parameter.name for parameter in defaulted_parameters
            )
            warnings.warn(
                f"Pipeline step {label} defines positional defaults ({defaulted_parameter_names}), "
                f"but Pipeline ignores positional defaults for dispatch to avoid ambiguity with tuple-valued "
                f"pipeline outputs. Validation will treat this operator as requiring "
                f"{len(parameters)} positional pipeline inputs.",
                warning_type,
                stacklevel=4,
            )

    return parameters


def validate_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    argument_label: str,
    error_type: type[Exception],
) -> inspect.Parameter:
    try:
        signature = _inspect_callable_signature(callable_)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} must expose an inspectable call signature because "
            f"{argument_label} is passed by position"
        ) from exc
    parameters = tuple(signature.parameters.values())

    input_parameter = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
        ),
        None,
    )
    if input_parameter is None:
        raise error_type(
            f"{label} must accept at least one positional value because "
            f"{argument_label} is passed by position"
        )

    unsupported_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind not in _POSITIONAL_VALUE_PARAMETER_KINDS
    )
    if unsupported_parameters:
        raise error_type(
            f"{label} uses non-positional parameters "
            f"({_format_parameter_descriptions(unsupported_parameters)}), "
            f"but {argument_label} is passed by position. Use only positional parameters."
        )

    return input_parameter


def validate_unary_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    argument_label: str,
    error_type: type[Exception],
) -> inspect.Parameter:
    input_parameter = validate_callable_signature(
        callable_,
        label=label,
        argument_label=argument_label,
        error_type=error_type,
    )
    try:
        inspect.signature(callable_).bind(object())
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} cannot be called with {argument_label}: {exc}"
        ) from exc
    return input_parameter


def validate_nullary_callable_signature(
    callable_: Callable[..., Any],
    *,
    label: str,
    error_type: type[Exception],
) -> None:
    try:
        signature = _inspect_callable_signature(callable_)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"{label} must expose an inspectable call signature because it is invoked without arguments"
        ) from exc
    parameters = tuple(signature.parameters.values())
    if parameters:
        raise error_type(
            f"{label} must define no parameters because it is invoked without arguments, but declares "
            f"({_format_parameter_descriptions(parameters)}). "
            f"Use a zero-argument callable with any configuration pre-bound."
        )


def _inspect_callable_signature(
    callable_: Callable[..., Any],
) -> inspect.Signature:
    return inspect.signature(callable_)


def _resolve_callable_hints(callable_: Callable[..., Any]) -> dict[str, Any]:
    return get_type_hints(_resolve_callable_hints_target(callable_))


def _resolve_callable_hints_target(callable_: Callable[..., Any]) -> Any:
    if inspect.isclass(callable_):
        return getattr(callable_, "__init__", callable_)
    if (
        inspect.isfunction(callable_)
        or inspect.ismethod(callable_)
        or inspect.isbuiltin(callable_)
        or inspect.ismethoddescriptor(callable_)
    ):
        return callable_
    return getattr(callable_, "__call__", callable_)


def _resolve_method_signature_annotations(
    owner: type,
    member: str,
) -> CallableSignatureAnnotations | None:
    try:
        method = getattr(owner, member)
    except AttributeError:
        return None
    if not callable(method):
        return None

    annotations = resolve_callable_signature_annotations(method)
    if not annotations.is_inspectable:
        return None
    return annotations


def _strip_member_receiver(
    owner: type,
    member: str,
    annotations: CallableSignatureAnnotations,
) -> CallableSignatureAnnotations | None:
    try:
        raw_member = inspect.getattr_static(owner, member)
    except AttributeError:
        return None
    if isinstance(raw_member, (staticmethod, classmethod)):
        return annotations
    return CallableSignatureAnnotations(
        annotations.parameters[1:],
        annotations.return_annotation,
        annotations.is_inspectable,
    )


def _parameter_defaults_match(source_default: Any, target_default: Any) -> bool:
    if (
        source_default is inspect.Parameter.empty
        or target_default is inspect.Parameter.empty
    ):
        return source_default is target_default
    if source_default is target_default:
        return True
    if type(source_default) is not type(target_default):
        return False
    try:
        return bool(source_default == target_default)
    except Exception:
        return False


def _format_parameter_descriptions(parameters: tuple[inspect.Parameter, ...]) -> str:
    return ", ".join(
        f"{parameter.name} ({parameter.kind.name.lower().replace('_', ' ')})"
        for parameter in parameters
    )
