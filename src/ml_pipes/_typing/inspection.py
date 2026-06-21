from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin, get_type_hints

from .annotation import (
    _MISSING_ANNOTATION,
    combine_annotation_options,
    describe_annotation,
    is_typed_dict_annotation,
    is_union_annotation,
    is_unknown_annotation,
)
from .signatures import _POSITIONAL_VALUE_PARAMETER_KINDS


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


def resolve_callable_annotations(
    callable_: Callable[..., Any],
) -> CallableAnnotations:
    positional_parameters = _resolve_positional_value_parameters(callable_)
    if positional_parameters is None:
        return CallableAnnotations((), None)

    try:
        hints = _resolve_callable_hints(callable_)
    except (TypeError, ValueError):
        return CallableAnnotations(
            tuple(None for _ in positional_parameters),
            None,
        )

    return_annotation = callable_ if inspect.isclass(callable_) else hints.get("return")
    return CallableAnnotations(
        tuple(hints.get(parameter.name) for parameter in positional_parameters),
        return_annotation,
    )


def probe_callable(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> inspect.BoundArguments:
    return inspect.signature(callable_).bind(*args, **kwargs)


def _resolve_positional_value_parameters(
    callable_: Callable[..., Any],
) -> tuple[inspect.Parameter, ...] | None:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return None

    return tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in _POSITIONAL_VALUE_PARAMETER_KINDS
    )


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
    if is_unknown_annotation(annotation):
        raise AttributeInspectionError(annotation, attribute, "owner annotation is unknown")

    if annotation in {None, type(None)}:
        raise MissingAttributeError(annotation, attribute)

    if is_union_annotation(annotation):
        resolved_options: list[Any] = []
        saw_missing_annotation = False
        for option in get_args(annotation):
            option_annotation = resolve_attribute_annotation(option, attribute)
            if option_annotation is _MISSING_ANNOTATION:
                saw_missing_annotation = True
                continue
            resolved_options.append(option_annotation)
        if not resolved_options or saw_missing_annotation:
            return _MISSING_ANNOTATION
        return combine_annotation_options(*resolved_options)

    if is_typed_dict_annotation(annotation):
        try:
            hints = get_type_hints(annotation)
        except (NameError, TypeError, ValueError) as exc:
            raise AttributeInspectionError(annotation, attribute, "typed dict annotations are unavailable") from exc
        if attribute not in hints:
            raise MissingTypedDictKeyError(annotation, attribute)
        return hints[attribute]

    override = _resolve_attribute_override(annotation, attribute)
    if override is not _MISSING_ANNOTATION:
        return override

    owner = _resolve_annotation_owner(annotation)
    if owner is None:
        raise AttributeInspectionError(annotation, attribute, "owner annotation is not inspectable")

    descriptor = getattr(owner, attribute, _MISSING_ANNOTATION)
    if isinstance(descriptor, property):
        if descriptor.fget is None:
            raise MissingAttributeError(annotation, attribute)
        try:
            hints = get_type_hints(descriptor.fget)
        except (NameError, TypeError, ValueError) as exc:
            raise AttributeInspectionError(annotation, attribute, "property annotations are unavailable") from exc
        return hints.get("return", _MISSING_ANNOTATION)

    try:
        hints = get_type_hints(owner)
    except (NameError, TypeError, ValueError) as exc:
        raise AttributeInspectionError(annotation, attribute, "attribute annotations are unavailable") from exc
    if attribute in hints:
        return hints[attribute]
    if descriptor is not _MISSING_ANNOTATION:
        return _MISSING_ANNOTATION
    raise MissingAttributeError(annotation, attribute)


def _resolve_annotation_owner(annotation: Any) -> type | None:
    origin = get_origin(annotation)
    if isinstance(origin, type):
        return origin
    if isinstance(annotation, type):
        return annotation
    return None


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
