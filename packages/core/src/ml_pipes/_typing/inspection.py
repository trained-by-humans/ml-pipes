from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, get_args, get_type_hints

from ml_pipes._typing.annotation import (
    _MISSING_ANNOTATION,
    _resolve_annotation_owner,
    build_union_annotation_from_options,
    describe_annotation,
    is_typed_dict_annotation,
    is_union_annotation,
    is_unknown_annotation,
)


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
class AttributeAnnotationInfo:
    annotation: Any
    is_writable: bool
    write_annotation: Any = _MISSING_ANNOTATION


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
        if _is_readonly_class_field_descriptor(owner, descriptor):
            return AttributeAnnotationInfo(hint, False)
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


def _is_readonly_class_field_descriptor(owner: type, descriptor: Any) -> bool:
    if descriptor is _MISSING_ANNOTATION:
        return False

    descriptor_type = type(descriptor)
    return (
        issubclass(owner, tuple)
        and descriptor_type.__module__ in {"collections", "_collections"}
        and descriptor_type.__name__ == "_tuplegetter"
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
