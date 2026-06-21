from typing import Any, TypedDict

import pytest

from ml_pipes._typing.annotation import _MISSING_ANNOTATION
from ml_pipes._typing.inspection import (
    AttributeInspectionError,
    MissingAttributeError,
    MissingTypedDictKeyError,
    resolve_attribute_annotation,
)


class _AnnotatedAttributeOwner:
    value: int


class _UntypedAttributeOwner:
    value = 1


class _PropertyAttributeOwner:
    @property
    def value(self) -> str:
        return ""


class _UntypedPropertyAttributeOwner:
    @property
    def value(self):
        return ""


class _BrokenAttributeOwner:
    value: "MissingType"


class _TypedDictAttributeOwner(TypedDict):
    value: int


class _BrokenTypedDictAttributeOwner(TypedDict):
    value: "MissingType"


@pytest.mark.parametrize(
    ("annotation", "attribute", "expected"),
    [
        pytest.param(_AnnotatedAttributeOwner, "value", int, id="annotated-attribute"),
        pytest.param(_PropertyAttributeOwner, "value", str, id="annotated-property"),
        pytest.param(_TypedDictAttributeOwner, "value", int, id="typed-dict-key"),
        pytest.param(_UntypedAttributeOwner, "value", _MISSING_ANNOTATION, id="untyped-attribute"),
        pytest.param(_UntypedPropertyAttributeOwner, "value", _MISSING_ANNOTATION, id="untyped-property"),
        pytest.param(
            _AnnotatedAttributeOwner | _UntypedAttributeOwner,
            "value",
            _MISSING_ANNOTATION,
            id="union-with-missing-annotation",
        ),
    ],
)
def test_resolve_attribute_annotation(annotation: Any, attribute: str, expected: Any) -> None:
    result = resolve_attribute_annotation(annotation, attribute)
    if expected is _MISSING_ANNOTATION:
        assert result is _MISSING_ANNOTATION
        return
    assert result == expected


def test_resolve_attribute_annotation_raises_for_missing_attribute() -> None:
    with pytest.raises(MissingAttributeError, match="has no attribute"):
        resolve_attribute_annotation(_AnnotatedAttributeOwner, "missing")


def test_resolve_attribute_annotation_raises_for_missing_typed_dict_key() -> None:
    with pytest.raises(MissingTypedDictKeyError, match="has no key"):
        resolve_attribute_annotation(_TypedDictAttributeOwner, "missing")


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(_BrokenAttributeOwner, id="broken-owner"),
        pytest.param(_BrokenTypedDictAttributeOwner, id="broken-typed-dict"),
    ],
)
def test_resolve_attribute_annotation_raises_for_unavailable_inspection(annotation: Any) -> None:
    with pytest.raises(AttributeInspectionError, match="annotations are unavailable"):
        resolve_attribute_annotation(annotation, "value")
