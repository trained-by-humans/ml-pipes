from collections.abc import Collection, Hashable, Iterable, Mapping, MutableSequence, Sequence
from typing import Any, Generic, Iterable as TypingIterable, Mapping as TypingMapping, TypeVar

import pytest

from ml_pipes._typing.annotation import (
    _annotation_shape,
    is_generic_indexable_annotation,
    is_generic_writable_indexable_annotation,
    is_mutable_sequence_annotation,
    is_assignable,
    normalize_published_annotation,
    remove_none_annotation_options,
    tighten_annotation,
)

_T = TypeVar("_T")


class _Box(Generic[_T]):
    pass


class _Indexable:
    def __getitem__(self, index: int) -> object:
        return index


class _WritableIndexable(_Indexable):
    def __setitem__(self, index: int, value: object) -> None:
        pass


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(type(None), None, id="none-type"),
        pytest.param(int | None, int, id="optional-int"),
        pytest.param(str | int | None, str | int, id="optional-union"),
    ],
)
def test_remove_none_annotation_options(annotation: Any, expected: Any) -> None:
    assert remove_none_annotation_options(annotation) == expected


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(list, (list, (Any,)), id="list"),
        pytest.param(list[int], (list, (int,)), id="list[int]"),
        pytest.param(Iterable, (Iterable, (Any,)), id="Iterable"),
        pytest.param(TypingIterable, (Iterable, (Any,)), id="typing.Iterable"),
        pytest.param(Iterable[int], (Iterable, (int,)), id="Iterable[int]"),
        pytest.param(Mapping, (Mapping, (Any, Any)), id="Mapping"),
        pytest.param(TypingMapping, (Mapping, (Any, Any)), id="typing.Mapping"),
        pytest.param(Mapping[str, int], (Mapping, (str, int)), id="Mapping[str,int]"),
        pytest.param(tuple, (tuple, (Any, Ellipsis)), id="tuple"),
        pytest.param(tuple[int, ...], (tuple, (int, Ellipsis)), id="tuple[int,...]"),
    ],
)
def test_annotation_shape_restores_missing_generic_arguments(annotation: Any, expected: Any) -> None:
    assert _annotation_shape(annotation) == expected


def test_annotation_shape_rejects_unsupported_bare_generic() -> None:
    with pytest.raises(ValueError, match="Unsupported bare generic annotation"):
        _annotation_shape(_Box)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(list[Any], list, id="list[Any]-to-list"),
        pytest.param(list[int], list, id="list[int]-to-list"),
        pytest.param(Iterable[Any], Iterable, id="Iterable[Any]-to-Iterable"),
        pytest.param(Iterable[int], Iterable, id="Iterable[int]-to-Iterable"),
        pytest.param(Mapping[str, int], Mapping, id="Mapping[str,int]-to-Mapping"),
        pytest.param(tuple[int, ...], tuple, id="tuple[int,...]-to-tuple"),
    ],
)
def test_is_assignable_accepts_parameterized_generic_for_bare_target(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(list, list[int], id="list-to-list[int]"),
        pytest.param(Iterable, Iterable[int], id="Iterable-to-Iterable[int]"),
        pytest.param(Mapping, Mapping[str, int], id="Mapping-to-Mapping[str,int]"),
        pytest.param(tuple, tuple[int, ...], id="tuple-to-tuple[int,...]"),
    ],
)
def test_is_assignable_accepts_bare_generic_for_parameterized_target(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(list[int], object, id="list[int]-to-object"),
        pytest.param(tuple[int, int], Hashable, id="tuple[int,int]-to-Hashable"),
    ],
)
def test_is_assignable_accepts_parameterized_generic_for_concrete_supertype(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(str, Iterable, id="str-to-Iterable"),
        pytest.param(bytes, Iterable, id="bytes-to-Iterable"),
        pytest.param(range, Iterable, id="range-to-Iterable"),
        pytest.param(str, Sequence, id="str-to-Sequence"),
        pytest.param(bytes, Sequence, id="bytes-to-Sequence"),
        pytest.param(range, Collection, id="range-to-Collection"),
    ],
)
def test_is_assignable_accepts_concrete_iterable_subtype_for_default_generic_target(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(str, Iterable[int], id="str-to-Iterable[int]"),
        pytest.param(bytes, Sequence[int], id="bytes-to-Sequence[int]"),
        pytest.param(range, Collection[str], id="range-to-Collection[str]"),
    ],
)
def test_is_assignable_rejects_concrete_iterable_subtype_for_typed_generic_target(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert not is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(list, list[Any], id="list"),
        pytest.param(Iterable, Iterable[Any], id="Iterable"),
        pytest.param(Mapping, Mapping[Any, Any], id="Mapping"),
        pytest.param(tuple, tuple[Any, ...], id="tuple"),
    ],
)
def test_normalize_published_annotation_restores_bare_generics(annotation: Any, expected: Any) -> None:
    assert normalize_published_annotation(annotation) == expected


@pytest.mark.parametrize(
    ("current_annotation", "candidate_annotation", "expected"),
    [
        pytest.param(list, list[int], list[int], id="list-with-list[int]"),
        pytest.param(Iterable, Iterable[int], Iterable[int], id="Iterable-with-Iterable[int]"),
        pytest.param(Mapping, Mapping[str, int], Mapping[str, int], id="Mapping-with-Mapping[str,int]"),
        pytest.param(tuple, tuple[int, ...], tuple[int, ...], id="tuple-with-tuple[int,...]"),
        pytest.param(list[int], list, list[int], id="list[int]-with-list"),
        pytest.param(Iterable[int], Iterable, Iterable[int], id="Iterable[int]-with-Iterable"),
        pytest.param(Mapping[str, int], Mapping, Mapping[str, int], id="Mapping[str,int]-with-Mapping"),
        pytest.param(tuple[int, ...], tuple, tuple[int, ...], id="tuple[int,...]-with-tuple"),
    ],
)
def test_tighten_annotation_restores_missing_generic_arguments(
    current_annotation: Any,
    candidate_annotation: Any,
    expected: Any,
) -> None:
    assert tighten_annotation(current_annotation, candidate_annotation) == expected


@pytest.mark.parametrize(
    "candidate_annotation",
    [
        pytest.param(Sequence[int | str], id="Sequence"),
        pytest.param(Iterable[int | str], id="Iterable"),
        pytest.param(Collection[int | str], id="Collection"),
    ],
)
def test_tighten_annotation_preserves_fixed_tuple_shape_against_sequence_supertypes(
    candidate_annotation: Any,
) -> None:
    assert tighten_annotation(tuple[int, str], candidate_annotation) == tuple[int, str]


def test_tighten_annotation_applies_sequence_constraint_per_fixed_tuple_item() -> None:
    assert tighten_annotation(tuple[Any, str], Sequence[int | str]) == tuple[int | str, str]


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation"),
    [
        pytest.param(_Box, _Box[int], id="source"),
        pytest.param(_Box[int], _Box, id="target"),
    ],
)
def test_is_assignable_rejects_unsupported_bare_generic(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    with pytest.raises(ValueError, match="Unsupported bare generic annotation"):
        is_assignable(source_annotation, target_annotation)


def test_normalize_published_annotation_rejects_unsupported_bare_generic() -> None:
    with pytest.raises(ValueError, match="Unsupported bare generic annotation"):
        normalize_published_annotation(_Box)


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(list[int], True, id="list"),
        pytest.param(MutableSequence[int], True, id="MutableSequence"),
        pytest.param(Sequence[int], False, id="Sequence"),
        pytest.param(tuple[int, ...], False, id="tuple"),
    ],
)
def test_is_mutable_sequence_annotation(annotation: Any, expected: bool) -> None:
    assert is_mutable_sequence_annotation(annotation) is expected


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(_Indexable, True, id="custom"),
        pytest.param(_WritableIndexable, True, id="writable-custom"),
        pytest.param(str, False, id="str"),
        pytest.param(dict[str, int], False, id="mapping"),
    ],
)
def test_is_generic_indexable_annotation(annotation: Any, expected: bool) -> None:
    assert is_generic_indexable_annotation(annotation) is expected


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(_Indexable, False, id="custom"),
        pytest.param(_WritableIndexable, True, id="writable-custom"),
        pytest.param(str, False, id="str"),
        pytest.param(dict[str, int], False, id="mapping"),
    ],
)
def test_is_generic_writable_indexable_annotation(annotation: Any, expected: bool) -> None:
    assert is_generic_writable_indexable_annotation(annotation) is expected
