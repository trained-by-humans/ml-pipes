from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from ml_pipes._typing.annotation import _annotation_shape, is_assignable, tighten_annotation


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(list, None, id="list"),
        pytest.param(list[int], (list, (int,)), id="list[int]"),
        pytest.param(Iterable, None, id="Iterable"),
        pytest.param(Iterable[int], (Iterable, (int,)), id="Iterable[int]"),
        pytest.param(Mapping, None, id="Mapping"),
        pytest.param(Mapping[str, int], (Mapping, (str, int)), id="Mapping[str,int]"),
        pytest.param(tuple, None, id="tuple"),
        pytest.param(tuple[int, ...], (tuple, (int, Ellipsis)), id="tuple[int,...]"),
    ],
)
def test_annotation_shape_keeps_bare_generics_unshaped(annotation: Any, expected: Any) -> None:
    assert _annotation_shape(annotation) == expected


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
def test_is_assignable_rejects_bare_generic_for_parameterized_target(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert not is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("current_annotation", "candidate_annotation", "expected"),
    [
        pytest.param(list, list[int], list, id="list-with-list[int]"),
        pytest.param(Iterable, Iterable[int], Iterable, id="Iterable-with-Iterable[int]"),
        pytest.param(Mapping, Mapping[str, int], Mapping, id="Mapping-with-Mapping[str,int]"),
        pytest.param(tuple, tuple[int, ...], tuple, id="tuple-with-tuple[int,...]"),
        pytest.param(list[int], list, list[int], id="list[int]-with-list"),
        pytest.param(Iterable[int], Iterable, Iterable[int], id="Iterable[int]-with-Iterable"),
        pytest.param(Mapping[str, int], Mapping, Mapping[str, int], id="Mapping[str,int]-with-Mapping"),
        pytest.param(tuple[int, ...], tuple, tuple[int, ...], id="tuple[int,...]-with-tuple"),
    ],
)
def test_tighten_annotation_preserves_current_bare_generic_policy(
    current_annotation: Any,
    candidate_annotation: Any,
    expected: Any,
) -> None:
    assert tighten_annotation(current_annotation, candidate_annotation) == expected
