from collections.abc import Iterable, Mapping
from typing import Any, Generic, Iterable as TypingIterable, Mapping as TypingMapping, TypeVar

import pytest

from ml_pipes._typing.annotation import (
    _annotation_shape,
    is_assignable,
    normalize_published_annotation,
    tighten_annotation,
)

_T = TypeVar("_T")


class _Box(Generic[_T]):
    pass


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
