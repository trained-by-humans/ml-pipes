from collections.abc import (
    Collection,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from typing import (
    Any,
    Generic,
    Iterable as TypingIterable,
    Mapping as TypingMapping,
    MutableMapping as TypingMutableMapping,
    MutableSequence as TypingMutableSequence,
    MutableSet as TypingMutableSet,
    Protocol,
    TypedDict,
    TypeVar,
)

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
_LegacySelf = TypeVar("Self")
_SourceBoxT = TypeVar("_SourceBoxT")
_InheritedBoxT = TypeVar("_InheritedBoxT")

try:
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self


class _Box(Generic[_T]):
    pass


class _GenericProtocolBox(Generic[_SourceBoxT]):
    value: _SourceBoxT

    def get(self) -> _SourceBoxT:
        raise NotImplementedError


class _InheritedGenericProtocolBox(
    Generic[_InheritedBoxT],
    _GenericProtocolBox[_InheritedBoxT],
):
    pass


class _ClosedIntProtocolBox(_GenericProtocolBox[int]):
    pass


class _ClosedStringProtocolBox(_GenericProtocolBox[str]):
    pass


class _IntValueProtocol(Protocol):
    value: int


class _IntGetterProtocol(Protocol):
    def get(self) -> int:
        ...


class _TypedDictIntValuePayload(TypedDict):
    value: int


class _Indexable:
    def __getitem__(self, index: int) -> object:
        return index


class _WritableIndexable(_Indexable):
    def __setitem__(self, index: int, value: object) -> None:
        pass


class _FilterableLabelsProtocol(Protocol):
    labels: Sequence[int]

    def filter(self, mask: Sequence[bool]) -> Self:
        ...


class _PredictionBase:
    def filter(self, mask: Sequence[bool]) -> Self:
        return self


class _IntLabelsPrediction(_PredictionBase):
    labels: Sequence[int]


class _IntLabelsPredictionChild(_IntLabelsPrediction):
    pass


class _StringLabelsPrediction(_PredictionBase):
    labels: Sequence[str]


class _IntReturningFilterPrediction:
    labels: Sequence[int]

    def filter(self, mask: Sequence[bool]) -> int:
        return 0


class _PredictionWithoutLabels(_PredictionBase):
    pass


class _PredictionWithRequiredFilterLimit(_PredictionBase):
    labels: Sequence[int]

    def filter(self, mask: Sequence[bool], *, limit: int) -> Self:
        del limit
        return self


class _LegacySelfFilterableProtocol(Protocol):
    labels: Sequence[int]

    def filter(self, mask: Sequence[bool]) -> _LegacySelf:
        ...


class _LegacySelfPredictionBase:
    def filter(self, mask: Sequence[bool]) -> _LegacySelf:
        return self


class _LegacySelfPreservingPrediction(_LegacySelfPredictionBase):
    labels: Sequence[int]


class _LegacySelfIntReturningPrediction:
    labels: Sequence[int]

    def filter(self, mask: Sequence[bool]) -> int:
        return 0


class _WritableObjectValueProtocol(Protocol):
    value: object


class _ReadonlyObjectValueProtocol(Protocol):
    @property
    def value(self) -> object:
        ...


class _IntValueField:
    value: int


class _ReadonlyIntValueProperty:
    @property
    def value(self) -> int:
        return 1


class _ObjectValueWithIntSetter:
    @property
    def value(self) -> object:
        return 1

    @value.setter
    def value(self, value: int) -> None:
        del value


class _IntValueWithObjectSetter:
    @property
    def value(self) -> int:
        return 1

    @value.setter
    def value(self, value: object) -> None:
        del value


class _KeywordLimitFilterProtocol(Protocol):
    def filter(self, mask: Sequence[bool], *, limit: int) -> Self:
        ...


class _KeywordLimitFilterWithRequiredLimit:
    def filter(self, mask: Sequence[bool], *, limit: int) -> Self:
        del mask, limit
        return self


class _KeywordLimitFilterWithOptionalLimit:
    def filter(self, mask: Sequence[bool], *, limit: int = 0) -> Self:
        del mask, limit
        return self


class _KeywordLimitFilterWithoutLimit:
    def filter(self, mask: Sequence[bool]) -> Self:
        del mask
        return self


class _MixerProtocol(Protocol):
    def mix(self, left: int, right: str) -> Self:
        ...


class _MixerWithMatchingOrder:
    def mix(self, left: int, right: str) -> Self:
        del left, right
        return self


class _MixerWithReorderedParameters:
    def mix(self, right: str, left: int) -> Self:
        del left, right
        return self


class _StaticBuildProtocol(Protocol):
    @staticmethod
    def build(value: int) -> str:
        ...


class _StaticBuildWithIntArg:
    @staticmethod
    def build(value: int) -> str:
        return str(value)


class _StaticBuildWithStringArg:
    @staticmethod
    def build(value: str) -> str:
        return value


class _ClassBuildProtocol(Protocol):
    @classmethod
    def build(cls, value: int) -> str:
        ...


class _ClassBuildWithIntArg:
    @classmethod
    def build(cls, value: int) -> str:
        del cls
        return str(value)


class _ClassBuildWithStringArg:
    @classmethod
    def build(cls, value: str) -> str:
        del cls
        return value


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
        pytest.param(MutableSequence, (MutableSequence, (Any,)), id="MutableSequence"),
        pytest.param(
            TypingMutableSequence,
            (MutableSequence, (Any,)),
            id="typing.MutableSequence",
        ),
        pytest.param(MutableMapping, (MutableMapping, (Any, Any)), id="MutableMapping"),
        pytest.param(
            TypingMutableMapping,
            (MutableMapping, (Any, Any)),
            id="typing.MutableMapping",
        ),
        pytest.param(MutableSet, (MutableSet, (Any,)), id="MutableSet"),
        pytest.param(TypingMutableSet, (MutableSet, (Any,)), id="typing.MutableSet"),
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
        pytest.param(MutableSequence, Sequence, id="MutableSequence-to-Sequence"),
        pytest.param(TypingMutableSequence, Sequence, id="typing.MutableSequence-to-Sequence"),
        pytest.param(MutableMapping, Mapping, id="MutableMapping-to-Mapping"),
        pytest.param(TypingMutableMapping, Mapping, id="typing.MutableMapping-to-Mapping"),
        pytest.param(MutableSet, AbstractSet, id="MutableSet-to-Set"),
        pytest.param(TypingMutableSet, AbstractSet, id="typing.MutableSet-to-Set"),
    ],
)
def test_is_assignable_accepts_bare_mutable_generic_aliases(
    source_annotation: Any,
    target_annotation: Any,
) -> None:
    assert is_assignable(source_annotation, target_annotation)


@pytest.mark.parametrize(
    ("source_annotation", "target_annotation", "expected_assignable"),
    [
        pytest.param(str, Iterable, True, id="str-to-Iterable"),
        pytest.param(bytes, Iterable, True, id="bytes-to-Iterable"),
        pytest.param(range, Iterable, True, id="range-to-Iterable"),
        pytest.param(str, Sequence, True, id="str-to-Sequence"),
        pytest.param(bytes, Sequence, True, id="bytes-to-Sequence"),
        pytest.param(range, Collection, True, id="range-to-Collection"),
        pytest.param(str, Iterable[int], False, id="str-to-Iterable[int]"),
        pytest.param(bytes, Sequence[int], False, id="bytes-to-Sequence[int]"),
        pytest.param(range, Collection[str], False, id="range-to-Collection[str]"),
    ],
)
def test_is_assignable_handles_concrete_iterable_subtype_targets(
    source_annotation: Any,
    target_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(source_annotation, target_annotation) is expected_assignable


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(list, list[Any], id="list"),
        pytest.param(Iterable, Iterable[Any], id="Iterable"),
        pytest.param(Mapping, Mapping[Any, Any], id="Mapping"),
        pytest.param(MutableSequence, MutableSequence[Any], id="MutableSequence"),
        pytest.param(TypingMutableSequence, MutableSequence[Any], id="typing.MutableSequence"),
        pytest.param(MutableMapping, MutableMapping[Any, Any], id="MutableMapping"),
        pytest.param(TypingMutableMapping, MutableMapping[Any, Any], id="typing.MutableMapping"),
        pytest.param(MutableSet, MutableSet[Any], id="MutableSet"),
        pytest.param(TypingMutableSet, MutableSet[Any], id="typing.MutableSet"),
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
    ("implementation_annotation", "expected_assignable"),
    [
        pytest.param(_IntLabelsPrediction, True, id="int-labels"),
        pytest.param(_IntLabelsPredictionChild, True, id="int-labels-subclass"),
        pytest.param(_StringLabelsPrediction, False, id="string-labels"),
        pytest.param(_IntReturningFilterPrediction, False, id="int-returning-filter"),
        pytest.param(
            _PredictionWithRequiredFilterLimit,
            False,
            id="required-filter-limit",
        ),
        pytest.param(_PredictionWithoutLabels, False, id="missing-labels"),
    ],
)
def test_is_assignable_checks_structural_protocol_implementation(
    implementation_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(
        implementation_annotation,
        _FilterableLabelsProtocol,
    ) is expected_assignable


def test_tighten_annotation_prefers_concrete_protocol_implementation() -> None:
    assert tighten_annotation(
        _FilterableLabelsProtocol,
        _IntLabelsPrediction,
    ) is _IntLabelsPrediction


@pytest.mark.parametrize(
    ("implementation_annotation", "protocol_annotation", "expected_assignable"),
    [
        pytest.param(
            _LegacySelfPreservingPrediction,
            _LegacySelfFilterableProtocol,
            True,
            id="legacy-self-preserving",
        ),
        pytest.param(
            _LegacySelfIntReturningPrediction,
            _LegacySelfFilterableProtocol,
            False,
            id="legacy-self-int-return",
        ),
        pytest.param(
            _IntValueField,
            _WritableObjectValueProtocol,
            False,
            id="int-field-for-writable-object",
        ),
        pytest.param(
            _ReadonlyIntValueProperty,
            _ReadonlyObjectValueProtocol,
            True,
            id="readonly-int-property-for-readonly-object",
        ),
        pytest.param(
            _ReadonlyIntValueProperty,
            _WritableObjectValueProtocol,
            False,
            id="readonly-int-property-for-writable-object",
        ),
        pytest.param(
            _ObjectValueWithIntSetter,
            _WritableObjectValueProtocol,
            False,
            id="object-value-with-int-setter",
        ),
        pytest.param(
            _IntValueWithObjectSetter,
            _WritableObjectValueProtocol,
            True,
            id="int-value-with-object-setter",
        ),
    ],
)
def test_is_assignable_handles_protocol_attribute_shapes(
    implementation_annotation: Any,
    protocol_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(
        implementation_annotation,
        protocol_annotation,
    ) is expected_assignable


@pytest.mark.parametrize(
    ("implementation_annotation", "expected_assignable"),
    [
        pytest.param(
            _KeywordLimitFilterWithRequiredLimit,
            True,
            id="required-limit",
        ),
        pytest.param(
            _KeywordLimitFilterWithOptionalLimit,
            True,
            id="optional-limit",
        ),
        pytest.param(
            _KeywordLimitFilterWithoutLimit,
            False,
            id="missing-limit",
        ),
    ],
)
def test_is_assignable_handles_protocol_keyword_only_method_parameter(
    implementation_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(
        implementation_annotation,
        _KeywordLimitFilterProtocol,
    ) is expected_assignable


@pytest.mark.parametrize(
    ("implementation_annotation", "expected_assignable"),
    [
        pytest.param(
            _MixerWithMatchingOrder,
            True,
            id="matching-order",
        ),
        pytest.param(
            _MixerWithReorderedParameters,
            False,
            id="reordered-parameters",
        ),
    ],
)
def test_is_assignable_preserves_protocol_method_positional_parameter_order(
    implementation_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(implementation_annotation, _MixerProtocol) is expected_assignable


@pytest.mark.parametrize(
    ("implementation_annotation", "protocol_annotation", "expected_assignable"),
    [
        pytest.param(
            _StaticBuildWithIntArg,
            _StaticBuildProtocol,
            True,
            id="static-build-int-arg",
        ),
        pytest.param(
            _StaticBuildWithStringArg,
            _StaticBuildProtocol,
            False,
            id="static-build-string-arg",
        ),
        pytest.param(
            _ClassBuildWithIntArg,
            _ClassBuildProtocol,
            True,
            id="class-build-int-arg",
        ),
        pytest.param(
            _ClassBuildWithStringArg,
            _ClassBuildProtocol,
            False,
            id="class-build-string-arg",
        ),
    ],
)
def test_is_assignable_handles_protocol_static_and_class_methods(
    implementation_annotation: Any,
    protocol_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(
        implementation_annotation,
        protocol_annotation,
    ) is expected_assignable


@pytest.mark.parametrize(
    ("implementation_annotation", "protocol_annotation", "expected_assignable"),
    [
        pytest.param(
            _GenericProtocolBox[int],
            _IntValueProtocol,
            True,
            id="generic-attr-int",
        ),
        pytest.param(
            _GenericProtocolBox[str],
            _IntValueProtocol,
            False,
            id="generic-attr-str",
        ),
        pytest.param(
            _GenericProtocolBox[int],
            _IntGetterProtocol,
            True,
            id="generic-method-int",
        ),
        pytest.param(
            _GenericProtocolBox[str],
            _IntGetterProtocol,
            False,
            id="generic-method-str",
        ),
        pytest.param(
            _InheritedGenericProtocolBox[int],
            _IntValueProtocol,
            True,
            id="inherited-generic-attr-int",
        ),
        pytest.param(
            _InheritedGenericProtocolBox[str],
            _IntValueProtocol,
            False,
            id="inherited-generic-attr-str",
        ),
        pytest.param(
            _InheritedGenericProtocolBox[int],
            _IntGetterProtocol,
            True,
            id="inherited-generic-method-int",
        ),
        pytest.param(
            _InheritedGenericProtocolBox[str],
            _IntGetterProtocol,
            False,
            id="inherited-generic-method-str",
        ),
        pytest.param(
            _ClosedIntProtocolBox,
            _IntGetterProtocol,
            True,
            id="closed-subclass-method-int",
        ),
        pytest.param(
            _ClosedStringProtocolBox,
            _IntGetterProtocol,
            False,
            id="closed-subclass-method-str",
        ),
    ],
)
def test_is_assignable_preserves_source_generic_specialization_for_protocols(
    implementation_annotation: Any,
    protocol_annotation: Any,
    expected_assignable: bool,
) -> None:
    assert is_assignable(
        implementation_annotation,
        protocol_annotation,
    ) is expected_assignable


def test_is_assignable_rejects_typed_dict_as_protocol_attribute_source() -> None:
    assert is_assignable(
        _TypedDictIntValuePayload,
        _IntValueProtocol,
    ) is False


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
