from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_type_hints

from ml_pipes._typing.annotation import (
    _MISSING_ANNOTATION,
    _resolve_annotation_owner,
    build_union_annotation_from_options,
    describe_annotation,
    is_typed_dict_annotation,
    is_union_annotation,
    is_unknown_annotation,
)
from ml_pipes._typing.signatures import _POSITIONAL_VALUE_PARAMETER_KINDS


class AttributeResolutionError(Exception):
    def __init__(self, annotation: Any, attribute: str, message: str) -> None:
        self.annotation = annotation
        self.attribute = attribute
        super().__init__(message)


class MissingAttributeError(AttributeResolutionError):
    def __init__(self, annotation: Any, attribute: str) -> None:
        super().__init__(
            annotation,
            attribute,
            f"{describe_annotation(annotation)} has no attribute {attribute!r}",
        )


class MissingTypedDictKeyError(AttributeResolutionError):
    def __init__(self, annotation: Any, key: str) -> None:
        super().__init__(
            annotation,
            key,
            f"{describe_annotation(annotation)} has no key {key!r}",
        )


class AttributeInspectionError(AttributeResolutionError):
    def __init__(self, annotation: Any, attribute: str, reason: str) -> None:
        self.reason = reason
        super().__init__(
            annotation,
            attribute,
            f"Cannot resolve attribute {attribute!r} for {describe_annotation(annotation)}: {reason}",
        )


@dataclass(frozen=True)
class CallableAnnotations:
    parameter_annotations: tuple[Any | None, ...]
    return_annotation: Any | None


@dataclass(frozen=True)
class CallableParameterAnnotation:
    parameter: inspect.Parameter
    annotation: Any | None


@dataclass(frozen=True)
class CallableSignatureAnnotations:
    parameters: tuple[CallableParameterAnnotation, ...]
    return_annotation: Any | None
    is_inspectable: bool


@dataclass(frozen=True)
class AttributeAnnotationInfo:
    annotation: Any
    is_writable: bool
    write_annotation: Any = _MISSING_ANNOTATION


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


def resolve_callable_annotations(
    callable_: Callable[..., Any],
) -> CallableAnnotations:
    signature_annotations = resolve_callable_signature_annotations(callable_)
    return CallableAnnotations(
        tuple(
            parameter.annotation
            for parameter in signature_annotations.parameters
            if parameter.parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
        ),
        signature_annotations.return_annotation,
    )


def probe_callable(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> inspect.BoundArguments:
    return inspect.signature(callable_).bind(*args, **kwargs)


def bind_method_call_parameter_names(
    callable_: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[object, str] | None:
    synthetic_self = object()
    try:
        bound_arguments = probe_callable(callable_, synthetic_self, *args, **kwargs)
    except (TypeError, ValueError):
        return None

    return {
        argument_value: parameter_name
        for parameter_name, argument_value in bound_arguments.arguments.items()
        if argument_value is not synthetic_self
    }


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


def resolve_attribute_annotation(annotation: Any, attribute: str) -> Any:
    return resolve_attribute_annotation_info(annotation, attribute).annotation


def resolve_attribute_annotation_info(annotation: Any, attribute: str) -> AttributeAnnotationInfo:
    if is_unknown_annotation(annotation):
        raise AttributeInspectionError(annotation, attribute, "owner annotation is unknown")

    if annotation in {None, type(None)}:
        raise MissingAttributeError(annotation, attribute)

    if is_union_annotation(annotation):
        resolved_options: list[Any] = []
        resolved_write_annotations: list[Any] = []
        is_writable = True
        saw_missing_annotation = False
        saw_missing_write_annotation = False
        for option in get_args(annotation):
            option_info = resolve_attribute_annotation_info(option, attribute)
            if option_info.annotation is _MISSING_ANNOTATION:
                saw_missing_annotation = True
                continue
            resolved_options.append(option_info.annotation)
            is_writable = is_writable and option_info.is_writable
            if option_info.is_writable:
                if option_info.write_annotation is _MISSING_ANNOTATION:
                    saw_missing_write_annotation = True
                else:
                    resolved_write_annotations.append(option_info.write_annotation)
        if not resolved_options or saw_missing_annotation:
            return AttributeAnnotationInfo(_MISSING_ANNOTATION, False)
        write_annotation = _MISSING_ANNOTATION
        if is_writable and not saw_missing_write_annotation:
            write_annotation = build_union_annotation_from_options(*resolved_write_annotations)
        return AttributeAnnotationInfo(
            build_union_annotation_from_options(*resolved_options),
            is_writable,
            write_annotation,
        )

    if is_typed_dict_annotation(annotation):
        try:
            hint = _resolve_class_field_annotation(annotation, attribute)
        except (NameError, TypeError, ValueError) as exc:
            raise AttributeInspectionError(annotation, attribute, "typed dict annotations are unavailable") from exc
        if hint is _MISSING_ANNOTATION:
            raise MissingTypedDictKeyError(annotation, attribute)
        return AttributeAnnotationInfo(hint, True, hint)

    override = _resolve_attribute_override(annotation, attribute)
    if override is not _MISSING_ANNOTATION:
        # Framework-level attribute overrides model a readable surface only.
        return AttributeAnnotationInfo(override, False)

    owner = _resolve_annotation_owner(annotation)
    if owner is None:
        raise AttributeInspectionError(annotation, attribute, "owner annotation is not inspectable")

    descriptor = getattr(owner, attribute, _MISSING_ANNOTATION)
    if isinstance(descriptor, property):
        if descriptor.fget is None:
            raise MissingAttributeError(annotation, attribute)
        try:
            getter_annotation = get_type_hints(descriptor.fget).get(
                "return",
                _MISSING_ANNOTATION,
            )
            setter_annotation = _resolve_property_setter_annotation(descriptor)
        except (NameError, TypeError, ValueError) as exc:
            raise AttributeInspectionError(annotation, attribute, "property annotations are unavailable") from exc
        return AttributeAnnotationInfo(
            getter_annotation,
            descriptor.fset is not None,
            setter_annotation,
        )

    try:
        hint = _resolve_class_field_annotation(owner, attribute)
    except (NameError, TypeError, ValueError) as exc:
        raise AttributeInspectionError(annotation, attribute, "attribute annotations are unavailable") from exc
    if hint is not _MISSING_ANNOTATION:
        return AttributeAnnotationInfo(hint, True, hint)
    if descriptor is not _MISSING_ANNOTATION:
        return AttributeAnnotationInfo(_MISSING_ANNOTATION, False)
    raise MissingAttributeError(annotation, attribute)


def _resolve_property_setter_annotation(descriptor: property) -> Any:
    if descriptor.fset is None:
        return _MISSING_ANNOTATION

    parameters = tuple(inspect.signature(descriptor.fset).parameters.values())
    if len(parameters) < 2:
        return _MISSING_ANNOTATION

    setter_value_name = parameters[1].name
    return get_type_hints(descriptor.fset).get(
        setter_value_name,
        _MISSING_ANNOTATION,
    )


def _resolve_class_field_annotation(owner: type, attribute: str) -> Any:
    resolved_owner_annotation = _locate_class_field_annotation(owner, attribute)
    if resolved_owner_annotation is _MISSING_ANNOTATION:
        return _MISSING_ANNOTATION

    declaring_owner, raw_annotation = resolved_owner_annotation
    globalns, localns = _owner_type_hint_namespaces(declaring_owner)
    proxy = type(
        "_OwnerAnnotationProxy",
        (),
        {
            "__module__": getattr(declaring_owner, "__module__", __name__),
            "__annotations__": {attribute: raw_annotation},
        },
    )
    return get_type_hints(proxy, globalns=globalns, localns=localns).get(
        attribute,
        _MISSING_ANNOTATION,
    )


def _locate_class_field_annotation(owner: type, attribute: str) -> tuple[type, Any] | object:
    for candidate in owner.__mro__:
        candidate_annotations = getattr(candidate, "__annotations__", None)
        if candidate_annotations is None or attribute not in candidate_annotations:
            continue
        return candidate, candidate_annotations[attribute]
    return _MISSING_ANNOTATION


def _owner_type_hint_namespaces(owner: type) -> tuple[dict[str, Any], dict[str, Any]]:
    module = inspect.getmodule(owner)
    globalns = vars(module) if module is not None else {}
    localns = dict(vars(owner))
    localns.setdefault(owner.__name__, owner)
    return globalns, localns


def _resolve_attribute_override(annotation: Any, attribute: str) -> Any:
    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION

    module = getattr(owner, "__module__", "")
    name = getattr(owner, "__name__", "")
    if attribute == "shape" and (
        (module.startswith("numpy") and name == "ndarray")
        or (module.startswith("torch") and name == "Tensor")
    ):
        return tuple[int, ...]
    return _MISSING_ANNOTATION
