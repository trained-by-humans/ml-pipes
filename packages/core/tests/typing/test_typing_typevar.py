from collections.abc import Iterable
from typing import Any, TypeVar

import pytest

from ml_pipes._typing.annotation import (
    _annotation_shape,
    _contains_ellipsize,
    _is_variadic,
    _is_variadic_shaped,
    _is_variadic_tuple,
    align_source_annotation_to_target_annotations,
    expand_annotation_parts,
    is_assignable,
    specialize_output_annotation_from_aligned_input_annotations,
    variadic_tuple_item_annotation,
)


class _Base:
    pass


class _Child(_Base):
    pass


class _Unrelated:
    pass


_T = TypeVar("_T", bound=_Base)
_U = TypeVar("_U")  # unbound
_ConstrainedT = TypeVar("_ConstrainedT", int, str)


def test_typevar_in_expected_accepts_bound_subclass():
    assert is_assignable(_Child, _T)


def test_typevar_in_expected_accepts_exact_bound():
    assert is_assignable(_Base, _T)


def test_typevar_in_expected_rejects_unrelated():
    assert not is_assignable(_Unrelated, _T)


def test_typevar_in_produced_accepts_when_bound_subclass_of_expected():
    assert is_assignable(_T, _Base)


def test_typevar_in_produced_rejects_when_expected_is_subtype_of_bound():
    assert not is_assignable(_T, _Child)


def test_typevar_in_produced_rejects_fully_unrelated():
    assert not is_assignable(_T, _Unrelated)


def test_unbound_typevar_in_expected_accepts_anything():
    assert is_assignable(int, _U)
    assert is_assignable(_Base, _U)


def test_unbound_typevar_in_produced_accepts_anything():
    assert is_assignable(_U, int)
    assert is_assignable(_U, _Base)


def test_generic_subtyping_accepts_list_as_iterable():
    assert is_assignable(list[int], Iterable[int])


def test_generic_subtyping_accepts_variadic_tuple_as_iterable():
    assert is_assignable(tuple[int, ...], Iterable[int])


def test_generic_covariance_accepts_child_list_as_base_iterable():
    assert is_assignable(list[_Child], Iterable[_Base])


def test_generic_invariance_rejects_child_list_as_base_list():
    assert not is_assignable(list[_Child], list[_Base])


def test_contains_ellipsize_detects_tuple_ellipsis_forms():
    assert _contains_ellipsize((int, Ellipsis))
    assert _contains_ellipsize((int, str, Ellipsis))
    assert _contains_ellipsize(tuple[int, ...])
    assert _contains_ellipsize(tuple[int, str, ...])
    assert not _contains_ellipsize((int, str))
    assert not _contains_ellipsize(tuple[int, str])


def test_variadic_shape_requires_single_item_and_ellipsis():
    assert _is_variadic_shaped((int, Ellipsis))
    assert not _is_variadic_shaped((int, str, Ellipsis))


def test_variadic_tuple_requires_single_item_and_ellipsis():
    assert _is_variadic_tuple(tuple[int, ...])
    assert not _is_variadic_tuple(tuple[int, str, ...])


def test_variadic_tuple_item_annotation_accepts_tuple_and_shape_forms():
    assert variadic_tuple_item_annotation(tuple[int, ...]) is int
    assert variadic_tuple_item_annotation((int, Ellipsis)) is int
    assert variadic_tuple_item_annotation(tuple[int, str]) is None
    assert variadic_tuple_item_annotation((int, str)) is None


def test_is_variadic_detects_shaped_and_tuple_variadics():
    assert _is_variadic((int, Ellipsis))
    assert _is_variadic(tuple[int, ...])
    assert not _is_variadic((int, str, Ellipsis))
    assert not _is_variadic(tuple[int, str, ...])


def test_annotation_shape_rejects_malformed_ellipsis_tuple_forms():
    with pytest.raises(ValueError, match="Malformed tuple annotation with ellipsis"):
        _annotation_shape((int, str, Ellipsis))

    with pytest.raises(ValueError, match="Malformed tuple annotation with ellipsis"):
        _annotation_shape(tuple[int, str, ...])


def test_expand_annotation_parts_rejects_malformed_ellipsis_tuple_forms():
    with pytest.raises(ValueError, match="Malformed tuple annotation with ellipsis"):
        expand_annotation_parts((int, str, Ellipsis))

    with pytest.raises(ValueError, match="Malformed tuple annotation with ellipsis"):
        expand_annotation_parts(tuple[int, str, ...])


def test_specialize_output_annotation_recursively_specializes_nested_output():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(_Child, (_T,))
    assert aligned_candidate_annotations == (_Child,)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (_T,),
        list[_T],
    ) == list[_Child]


def test_specialize_output_annotation_recursively_specializes_plain_tuple_output():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(_Child, (_T,))
    assert aligned_candidate_annotations == (_Child,)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (_T,),
        (_T, list[_T]),
    ) == (
        _Child,
        list[_Child],
    )


def test_specialize_output_annotation_merges_repeated_typevar_inputs():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(
        tuple[_Base, _Child],
        (_T, _T),
    )
    assert aligned_candidate_annotations == (_Base, _Child)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (_T, _T),
        _T,
    ) is _Base


def test_specialize_output_annotation_from_single_tuple_parameter():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(
        tuple[_Child, int],
        (tuple[_T, int],),
    )
    assert aligned_candidate_annotations == (tuple[_Child, int],)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (tuple[_T, int],),
        _T,
    ) is _Child


def test_specialize_output_annotation_through_generic_subtyping():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(
        list[int | None],
        (Iterable[_U | None],),
    )
    assert aligned_candidate_annotations == (list[int | None],)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (Iterable[_U | None],),
        list[_U],
    ) == list[int]


def test_specialize_output_annotation_does_not_bind_constrained_typevar_from_any_input():
    aligned_candidate_annotations = align_source_annotation_to_target_annotations(
        Any,
        (_ConstrainedT,),
    )
    assert aligned_candidate_annotations == (Any,)
    assert specialize_output_annotation_from_aligned_input_annotations(
        aligned_candidate_annotations,
        (_ConstrainedT,),
        _ConstrainedT,
    ) is _ConstrainedT
