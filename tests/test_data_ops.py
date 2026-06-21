from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from ml_pipes import (
    CollectItems,
    Distinct,
    DistinctBy,
    DropNull,
    FilterNotNull,
    Filter,
    LazyPerItem,
    Map,
    MapNotNull,
    MapValue,
    Pipeline,
    PerItem,
    SHORT_CIRCUIT,
    StreamItems,
    WrapMappingInObject,
    Take,
    TakeWhile,
    PipelineValidationError,
)


@dataclass
class Carrier:
    payload: dict[str, object] | None = None
    cleaned: str | None = None
    length: int | None = None


def _message(label: str, msg: str) -> dict[str, object]:
    return {"label": label, "msg": msg}


def _text_length(text: str) -> int:
    return len(text)


def _optional_text_length(text: str | None) -> int:
    return 0 if text is None else len(text)


def _text_length_or_none(text: str) -> int | None:
    if not text:
        return None
    return len(text)


def _has_min_length(text: str) -> bool:
    return len(text) >= 5


def _has_min_length_or_none(text: str | None) -> bool:
    return text is not None and len(text) >= 5


def _int_to_text(value: int) -> str:
    return str(value)


def _scale_int(value: int, *, factor: int = 1) -> int:
    return value * factor


def _scale_int_requires_factor(value: int, *, factor: int) -> int:
    return value * factor


def _add_two_ints(left: int, right: int) -> int:
    return left + right


def _text_has_min_length_with_floor(text: str, *, floor: int = 0) -> bool:
    return len(text) > floor


def _append_suffix(text: str, *, suffix: str = "") -> str:
    return text + suffix


class AcceptInt:
    def __call__(self, value: int) -> str:
        return str(value)


class AcceptString:
    def __call__(self, value: str) -> str:
        return value.upper()


class AcceptCarrier:
    def __call__(self, value: Carrier) -> str:
        return value.cleaned or ""


class AcceptIntList:
    def __call__(self, items: list[int]) -> str:
        return ",".join(str(item) for item in items)


class AcceptCarrierList:
    def __call__(self, items: list[Carrier]) -> int:
        return len(items)


class ClosableIterator:
    def __init__(self, values: list[int]):
        self._iterator = iter(values)
        self.closed = False

    def __iter__(self) -> ClosableIterator:
        return self

    def __next__(self) -> int:
        return next(self._iterator)

    def close(self) -> None:
        self.closed = True


@dataclass
class StrictCarrier:
    cleaned: str


@dataclass
class StrictMappedCarrier:
    cleaned: str
    length: int


@dataclass
class NoneOnlyCarrier:
    cleaned: None = None


@dataclass
class WrongPayloadCarrier:
    payload: str = ""


@dataclass
class NonHashableKeyCarrier:
    tags: list[str]


@dataclass
class Box:
    value: int


def _make_carrier() -> Carrier:
    return Carrier()


def _make_carrier_with_default_payload(payload: dict[str, object] | None = None) -> Carrier:
    return Carrier(payload=payload)


def _make_wrong_payload_carrier() -> WrongPayloadCarrier:
    return WrongPayloadCarrier()


def _box_int(value) -> Box:
    return Box(value)


def _box_int_or_none(value) -> Box | None:
    if value < 0:
        return None
    return Box(value)


def _always_none(value: int) -> None:
    return None


def _is_positive(value: int) -> bool:
    return value > 0


def _is_positive_or_none(value: int | None) -> bool:
    return value is None or value > 0


def _is_positive_without_return_annotation(value: int):
    return value > 0


def _is_positive_as_int(value: int) -> int:
    return value


def _hash_text(text: str) -> int:
    return len(text)


def _split_text(text: str) -> list[str]:
    return list(text)


def _lower_text(text: str) -> str:
    return text.lower()


def _hash_optional_text(text: str | None) -> int:
    return 0 if text is None else len(text)


def _short_circuit_on_two(value: int) -> int | object:
    if value == 2:
        return SHORT_CIRCUIT
    return value


def _multiply_by_ten(value: int) -> int:
    return value * 10


def _raise_on_two(value: int) -> int:
    if value == 2:
        raise ValueError("bad item")
    return value


def test_map_transforms_current_value() -> None:
    assert Map(lambda value: value * 2)(3) == 6


def test_map_passes_none_to_mapper_when_mapper_accepts_optional() -> None:
    assert Map(_optional_text_length)(None) == 0


def test_map_not_null_short_circuits_none_result() -> None:
    assert MapNotNull(_text_length_or_none)("") is SHORT_CIRCUIT
    assert MapNotNull(_text_length_or_none)("abc") == 3


def test_map_value_updates_selected_field_in_place_when_src_is_set() -> None:
    carrier = Carrier(payload={"msg": "  hello  "})

    result = MapValue(str.strip, source="payload.msg")(carrier)

    assert result is carrier
    assert carrier.payload == {"msg": "hello"}


def test_filter_applies_predicate_to_current_value() -> None:
    assert Filter(lambda value: value > 1)(2) == 2
    assert Filter(lambda value: value > 1)(1) is SHORT_CIRCUIT


def test_filter_passes_none_to_optional_aware_predicate() -> None:
    assert Filter(_is_positive_or_none)(None) is None


def test_filter_applies_predicate_to_selected_value() -> None:
    carrier = Carrier(payload={"msg": "hello"})

    assert Filter(lambda text: text == "hello", source="payload.msg")(carrier) is carrier
    assert Filter(lambda text: text == "bye", source="payload.msg")(carrier) is SHORT_CIRCUIT


def test_filter_validation_returns_surviving_output_type() -> None:
    contract = Pipeline([Filter(_is_positive)]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    assert contract.input_type == int
    assert contract.output_type == int


def test_filter_validation_keeps_unknown_output_unknown() -> None:
    contract = Pipeline([Filter(_is_positive)]).validate()

    assert contract.output_type == Any


def test_filter_validation_allows_non_optional_downstream_consumer() -> None:
    contract = Pipeline([Filter(_is_positive), AcceptInt()]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    assert contract.input_type == int
    assert contract.output_type == str


def test_filter_validation_requires_optional_aware_predicate_for_optional_input() -> None:
    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([Filter(_is_positive)]).validate(
            pipeline_input_type=int | None,
            strict=True,
        )

    contract = Pipeline([Filter(_is_positive_or_none)]).validate(
        pipeline_input_type=int | None,
        strict=True,
    )

    assert contract.input_type == int | None
    assert contract.output_type == int | None


def test_filter_validation_allows_predicate_without_return_annotation() -> None:
    contract = Pipeline([Filter(_is_positive_without_return_annotation)]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    assert contract.input_type == int
    assert contract.output_type == int


def test_filter_validation_rejects_non_bool_predicate_return_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="predicate return type expects"):
        Pipeline([Filter(_is_positive_as_int)]).validate(
            pipeline_input_type=int,
            strict=True,
        )


def test_filter_validation_checks_selected_source_type_against_predicate() -> None:
    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([Filter(_is_positive, source="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_filter_validation_rejects_missing_selector_even_for_untyped_predicate() -> None:
    with pytest.raises(PipelineValidationError, match="missing"):
        Pipeline([Filter(lambda value: True, source="missing")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_filter_rejects_predicate_with_non_positional_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        Filter(_text_has_min_length_with_floor)


def test_filter_validation_propagates_name_error_from_predicate_annotations() -> None:
    def broken(value: "MissingType") -> bool:
        return True

    with pytest.raises(NameError, match="MissingType"):
        Pipeline([Filter(broken)]).validate(
            pipeline_input_type=int,
        )


def test_map_validation_requires_optional_aware_mapper_for_optional_input() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([Map(_text_length)]).validate(
            pipeline_input_type=str | None,
            strict=True,
        )


def test_map_validation_accepts_optional_aware_mapper_for_optional_input() -> None:
    contract = Pipeline([Map(_optional_text_length)]).validate(
        pipeline_input_type=str | None,
        strict=True,
    )

    assert contract.input_type == str | None
    assert contract.output_type == int


def test_map_validation_propagates_name_error_from_mapper_annotations() -> None:
    def broken(value: "MissingType") -> int:
        return 1

    with pytest.raises(NameError, match="MissingType"):
        Pipeline([Map(broken), AcceptInt()]).validate(
            pipeline_input_type=int,
        )


def test_map_validation_rejects_mapper_without_usable_output_annotations() -> None:
    def untyped_length(text: str):
        return len(text)

    with pytest.raises(
        PipelineValidationError,
        match="Map fn must define a usable return type annotation",
    ):
        Pipeline([Map(untyped_length), AcceptString()]).validate(
            pipeline_input_type=str,
        )


def test_map_validation_accepts_mapper_without_input_annotation_when_upstream_is_concrete() -> None:
    contract = Pipeline([Map(_box_int)]).validate(
        pipeline_input_type=int,
    )

    assert contract.input_type == int
    assert contract.output_type == Box


def test_map_validation_requires_input_annotation_when_upstream_is_unknown() -> None:
    with pytest.raises(
        PipelineValidationError,
        match="Map fn must define a usable input type annotation",
    ):
        Pipeline([Map(_box_int)]).validate()


def test_map_rejects_mapper_with_non_positional_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        Map(_scale_int)


def test_map_rejects_mapper_with_required_keyword_only_parameter_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        Map(_scale_int_requires_factor)


def test_map_rejects_mapper_with_required_additional_positional_parameter_at_construction() -> None:
    with pytest.raises(TypeError, match="cannot be called"):
        Map(_add_two_ints)


def test_map_not_null_validation_preserves_concrete_output_for_non_optional_mapper() -> None:
    contract = Pipeline([MapNotNull(_text_length)]).validate(
        pipeline_input_type=str,
        strict=True,
    )

    assert contract.input_type == str
    assert contract.output_type == int


def test_map_not_null_validation_returns_surviving_mapped_type() -> None:
    contract = Pipeline([MapNotNull(_text_length_or_none)]).validate(
        pipeline_input_type=str,
        strict=True,
    )

    assert contract.input_type == str
    assert contract.output_type == int

    Pipeline([MapNotNull(_text_length_or_none), AcceptInt()]).validate(
        pipeline_input_type=str,
        strict=True,
    )


def test_map_not_null_validation_accepts_mapper_without_input_annotation_when_upstream_is_concrete() -> None:
    contract = Pipeline([MapNotNull(_box_int_or_none)]).validate(
        pipeline_input_type=int,
    )

    assert contract.input_type == int
    assert contract.output_type == Box


def test_map_not_null_validation_rejects_none_only_mapper_output() -> None:
    with pytest.raises(PipelineValidationError, match="cannot produce a non-None output"):
        Pipeline([MapNotNull(_always_none)]).validate(
            pipeline_input_type=int,
        )


def test_map_not_null_validation_requires_optional_aware_mapper_for_optional_input() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([MapNotNull(_text_length_or_none)]).validate(
            pipeline_input_type=str | None,
            strict=True,
        )

    Pipeline([MapNotNull(_optional_text_length), AcceptInt()]).validate(
        pipeline_input_type=str | None,
        strict=True,
    )


def test_map_value_validation_checks_selected_source_type_against_mapper() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([MapValue(_int_to_text, source="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_map_value_validation_requires_optional_aware_mapper_for_optional_selector() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([MapValue(_text_length, source="cleaned")]).validate(
            pipeline_input_type=Carrier,
            strict=True,
        )

    Pipeline([MapValue(_optional_text_length, source="cleaned", target="length")]).validate(
        pipeline_input_type=Carrier,
        strict=True,
    )


def test_map_value_validation_rejects_missing_selector_even_for_untyped_mapper() -> None:
    with pytest.raises(PipelineValidationError, match="missing"):
        Pipeline([MapValue(lambda value: value, source="missing")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_map_value_validation_rejects_incompatible_target_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="target"):
        Pipeline([MapValue(_text_length, source="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_map_value_rejects_mapper_with_non_positional_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        MapValue(_append_suffix, source="cleaned", target="length")


def test_map_value_validation_propagates_name_error_from_mapper_annotations() -> None:
    def broken(value: "MissingType") -> int:
        return 1

    with pytest.raises(NameError, match="MissingType"):
        Pipeline([MapValue(broken, source="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
        )


def test_filter_not_null_drops_missing_selector() -> None:
    carrier = Carrier(payload={})

    assert FilterNotNull(source="payload.msg")(carrier) is SHORT_CIRCUIT


def test_filter_not_null_keeps_present_selector() -> None:
    carrier = Carrier(payload={"msg": "hello"})

    assert FilterNotNull(source="payload.msg")(carrier) is carrier


def test_filter_not_null_validation_keeps_current_object_type() -> None:
    contract = Pipeline([FilterNotNull(source="payload.msg")]).validate(
        pipeline_input_type=Carrier,
        strict=True,
    )

    assert contract.input_type == Carrier
    assert contract.output_type == Carrier

    Pipeline([FilterNotNull(source="payload.msg"), AcceptCarrier()]).validate(
        pipeline_input_type=Carrier,
        strict=True,
    )


def test_filter_not_null_validation_rejects_none_only_source_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="cannot produce a non-None output"):
        Pipeline([FilterNotNull(source="cleaned")]).validate(
            pipeline_input_type=NoneOnlyCarrier,
            strict=True,
        )


def test_filter_not_null_validation_rejects_missing_selector() -> None:
    with pytest.raises(PipelineValidationError, match="missing"):
        Pipeline([FilterNotNull(source="missing")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_filter_raises_on_missing_selector() -> None:
    carrier = Carrier(payload={})

    with pytest.raises(TypeError, match="cannot resolve"):
        Filter(lambda text: True, source="payload.msg")(carrier)


def test_wrap_mapping_in_object_seeds_object_factory() -> None:
    result = WrapMappingInObject(target="payload", state_factory=_make_carrier)(_message("spam", "hello"))

    assert isinstance(result, Carrier)
    assert result.payload == {"label": "spam", "msg": "hello"}


def test_wrap_mapping_in_object_validation_returns_factory_output_when_input_is_unknown() -> None:
    contract = Pipeline([WrapMappingInObject(target="payload", state_factory=_make_carrier)]).validate()

    assert contract.output_type == Carrier


def test_wrap_mapping_in_object_rejects_factory_with_hidden_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="state_factory must define no parameters"):
        WrapMappingInObject(target="payload", state_factory=_make_carrier_with_default_payload)


def test_wrap_mapping_in_object_validation_propagates_name_error_from_factory_annotations() -> None:
    def broken_factory() -> "MissingType":
        return Carrier()

    with pytest.raises(NameError, match="MissingType"):
        Pipeline([WrapMappingInObject(target="payload", state_factory=broken_factory)]).validate(
            pipeline_input_type=dict[str, object],
        )


def test_wrap_mapping_in_object_validation_rejects_factory_without_usable_output_annotation() -> None:
    def untyped_factory():
        return Carrier()

    with pytest.raises(
        PipelineValidationError,
        match="WrapMappingInObject state_factory must define a usable return type annotation",
    ):
        Pipeline([WrapMappingInObject(target="payload", state_factory=untyped_factory)]).validate(
            pipeline_input_type=dict[str, object],
        )


def test_wrap_mapping_in_object_validation_rejects_incompatible_target_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="target"):
        Pipeline([WrapMappingInObject(target="payload", state_factory=_make_wrong_payload_carrier)]).validate(
            pipeline_input_type=dict[str, object],
            strict=True,
        )


def test_drop_null_short_circuits_none_current_value() -> None:
    assert DropNull()(None) is SHORT_CIRCUIT
    assert DropNull()(2) == 2


def test_drop_null_validation_uses_static_contract() -> None:
    contract = Pipeline([DropNull(), AcceptInt()]).validate(
        pipeline_input_type=int | None,
        strict=True,
    )

    assert contract.input_type == int | None
    assert contract.output_type == str


def test_drop_null_validation_rejects_none_only_input() -> None:
    with pytest.raises(PipelineValidationError, match="cannot produce a non-None output"):
        Pipeline([DropNull()]).validate(
            pipeline_input_type=None,
        )


def test_distinct_deduplicates_by_selected_value() -> None:
    items = [
        Carrier(payload=_message("spam", "one"), cleaned="dup"),
        Carrier(payload=_message("ham", "two"), cleaned="dup"),
        Carrier(payload=_message("ham", "three"), cleaned="unique"),
    ]

    result = Distinct(source="cleaned")(items)

    assert [item.payload["msg"] for item in result if item.payload is not None] == ["one", "three"]


def test_distinct_validation_accepts_existing_selector() -> None:
    Pipeline([Distinct(source="cleaned")]).validate(
        pipeline_input_type=list[StrictCarrier],
        strict=True,
    )


def test_distinct_validation_rejects_missing_selector() -> None:
    with pytest.raises(PipelineValidationError, match="missing"):
        Pipeline([Distinct(source="missing")]).validate(
            pipeline_input_type=list[StrictCarrier],
            strict=True,
        )


def test_distinct_validation_rejects_out_of_bounds_tuple_selector() -> None:
    with pytest.raises(PipelineValidationError, match="out of bounds"):
        Pipeline([Distinct(source=2)]).validate(
            pipeline_input_type=list[tuple[str, int]],
            strict=True,
        )


def test_distinct_by_deduplicates_by_computed_value() -> None:
    result = DistinctBy(_hash_text)(["a", "bb", "cc", "d"])

    assert result == ["a", "bb"]


def test_distinct_by_validation_checks_fn_input_type() -> None:
    Pipeline([DistinctBy(_hash_text)]).validate(
        pipeline_input_type=list[str],
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([DistinctBy(_is_positive)]).validate(
            pipeline_input_type=list[str],
            strict=True,
        )


def test_distinct_by_validation_requires_optional_aware_key_fn_for_optional_items() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([DistinctBy(_hash_text)]).validate(
            pipeline_input_type=list[str | None],
            strict=True,
        )

    Pipeline([DistinctBy(_hash_optional_text)]).validate(
        pipeline_input_type=list[str | None],
        strict=True,
    )


def test_distinct_by_validation_rejects_non_hashable_key_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="fn return type expects"):
        Pipeline([DistinctBy(_split_text)]).validate(
            pipeline_input_type=list[str],
            strict=True,
        )


def test_distinct_by_rejects_key_fn_with_non_positional_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        DistinctBy(_append_suffix)


def test_take_supports_iterables() -> None:
    items = (value for value in range(5))

    assert Take(2)(items) == [0, 1]


def test_take_supports_value_shaped_iterables() -> None:
    assert Take(2)("hello") == ["h", "e"]


def test_take_closes_iterator_when_it_short_circuits() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert Take(2)(items) == [0, 1]
    assert items.closed is True


def test_take_validation_uses_static_contract() -> None:
    contract = Pipeline([Take(2), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type == str


def test_take_validation_accepts_string_iterable_boundary() -> None:
    contract = Pipeline([Take(2)]).validate(
        pipeline_input_type=str,
        strict=True,
    )

    assert contract.input_type == str
    assert contract.output_type == list[str]


def test_take_rejects_scalar_boundary_during_validation() -> None:
    with pytest.raises(PipelineValidationError, match="iterable boundary"):
        Pipeline([Take(2)]).validate(
            pipeline_input_type=int,
            strict=True,
        )


def test_take_while_supports_iterables() -> None:
    assert TakeWhile(lambda value: value < 3)(range(5)) == [0, 1, 2]


def test_take_while_supports_value_shaped_iterables() -> None:
    assert TakeWhile(lambda value: value != "!")("hello!") == ["h", "e", "l", "l", "o"]


def test_take_while_closes_iterator_when_it_short_circuits() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert TakeWhile(lambda value: value < 2)(items) == [0, 1]
    assert items.closed is True


def test_take_while_validation_checks_predicate_input_type() -> None:
    Pipeline([TakeWhile(_is_positive)]).validate(
        pipeline_input_type=list[int],
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([TakeWhile(_has_min_length)]).validate(
            pipeline_input_type=list[int],
            strict=True,
        )


def test_take_while_rejects_predicate_with_non_positional_parameters_at_construction() -> None:
    with pytest.raises(TypeError, match="uses non-positional parameters"):
        TakeWhile(_text_has_min_length_with_floor)


def test_take_while_validation_allows_predicate_without_return_annotation() -> None:
    Pipeline([TakeWhile(_is_positive_without_return_annotation)]).validate(
        pipeline_input_type=list[int],
        strict=True,
    )


def test_take_while_validation_rejects_non_bool_predicate_return_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="predicate return type expects"):
        Pipeline([TakeWhile(_is_positive_as_int)]).validate(
            pipeline_input_type=list[int],
            strict=True,
        )


def test_take_while_rejects_scalar_boundary_during_validation() -> None:
    with pytest.raises(PipelineValidationError, match="iterable boundary"):
        Pipeline([TakeWhile(_is_positive)]).validate(
            pipeline_input_type=int,
            strict=True,
        )


def test_distinct_by_supports_value_shaped_iterables() -> None:
    assert DistinctBy(_lower_text)("AaBbA") == ["A", "B"]


def test_distinct_by_validation_accepts_string_iterable_boundary() -> None:
    contract = Pipeline([DistinctBy(_lower_text)]).validate(
        pipeline_input_type=str,
        strict=True,
    )

    assert contract.input_type == str
    assert contract.output_type == list[str]


def test_distinct_validation_rejects_non_hashable_selector_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="distinct key type expects"):
        Pipeline([Distinct(source="tags")]).validate(
            pipeline_input_type=list[NonHashableKeyCarrier],
            strict=True,
        )


@pytest.mark.parametrize("annotation", [set[int], frozenset[int], Iterator[int]])
def test_per_item_validation_accepts_generic_item_iterable_annotations(annotation: Any) -> None:
    contract = Pipeline(
        [
            PerItem(),
            _multiply_by_ten,
            CollectItems(),
            AcceptIntList(),
        ]
    ).validate(
        pipeline_input_type=annotation,
        strict=True,
    )

    assert contract.input_type == annotation
    assert contract.output_type == str


@pytest.mark.parametrize(
    ("opener", "closer"),
    [
        (PerItem, CollectItems),
        (LazyPerItem, StreamItems),
    ],
)
@pytest.mark.parametrize(
    "current",
    [
        {"id": "sms-00000", "label": "ham", "text": "hello"},
        "hello",
        b"hello",
    ],
)
def test_per_item_regions_reject_single_mapping_and_string_like_boundaries_at_runtime(
    opener: type[PerItem] | type[LazyPerItem],
    closer: type[CollectItems] | type[StreamItems],
    current: object,
) -> None:
    pipeline = Pipeline(
        [
            opener(),
            DropNull(),
            closer(),
        ]
    )

    with pytest.raises(TypeError, match=rf"{opener.__name__} requires an item iterable boundary"):
        pipeline(current)


@pytest.mark.parametrize(
    ("opener", "closer"),
    [
        (PerItem, CollectItems),
        (LazyPerItem, StreamItems),
    ],
)
@pytest.mark.parametrize(
    "annotation",
    [
        dict[str, object],
        str,
        bytes,
    ],
)
def test_per_item_regions_reject_single_mapping_and_string_like_boundaries_during_validation(
    opener: type[PerItem] | type[LazyPerItem],
    closer: type[CollectItems] | type[StreamItems],
    annotation: Any,
) -> None:
    with pytest.raises(PipelineValidationError, match=rf"{opener.__name__} requires an item iterable boundary"):
        Pipeline(
            [
                opener(),
                DropNull(),
                closer(),
            ]
        ).validate(
            pipeline_input_type=annotation,
            strict=True,
        )


def test_take_after_per_item_pipeline_reuse_is_stable() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _multiply_by_ten,
            CollectItems(),
            Take(2),
        ]
    )

    assert pipeline([1, 2, 3]) == [10, 20]
    assert pipeline([1, 2, 3]) == [10, 20]


def test_drop_null_inside_per_item_drops_none_items() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            DropNull(),
            CollectItems(),
        ]
    )

    assert pipeline([1, None, 2]) == [1, 2]


def test_take_after_per_item_limits_materialized_output() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _multiply_by_ten,
            CollectItems(),
            Take(2),
        ]
    )

    assert pipeline([0, 1, 2, 3]) == [0, 10]


def test_per_item_drops_short_circuited_items() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _short_circuit_on_two,
            _multiply_by_ten,
            CollectItems(),
        ]
    )

    assert pipeline([1, 2, 3]) == [10, 30]


def test_take_inside_per_item_is_rejected_as_collection_op() -> None:
    with pytest.raises(PipelineValidationError, match="iterable boundary"):
        Pipeline(
            [
                PerItem(),
                Take(2),
                CollectItems(),
            ]
        ).validate(
            pipeline_input_type=list[int],
            strict=True,
        )


def test_per_item_then_take_validates_against_collection_output() -> None:
    contract = Pipeline(
        [
            PerItem(),
            _multiply_by_ten,
            CollectItems(),
            Take(2),
            AcceptIntList(),
        ]
    ).validate(
        pipeline_input_type=list[int],
        strict=True,
    )

    assert contract.input_type == list[int]
    assert contract.output_type == str


def test_take_after_per_item_is_visible_in_parent_trace() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _multiply_by_ten,
            CollectItems(),
            Take(2),
        ]
    )

    result = pipeline.inspect([0, 1, 2, 3])

    assert [span.label for span in result.spans] == ["0:PerItem", "3:Take"]
    assert result.spans[0].attributes == {
        "seen": 4,
        "emitted": 4,
        "dropped": 0,
    }
    assert result.spans[0].output_value == [0, 10, 20, 30]
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == ["1:_multiply_by_ten"]
    assert result.spans[0].child_trace.spans[0].attributes == {
        "dropped": 0,
    }
    assert result.spans[1].output_value == [0, 10]


def test_lazy_per_item_only_processes_consumed_items() -> None:
    seen: list[int] = []

    def mark(value: int) -> int:
        seen.append(value)
        return value * 10

    pipeline = Pipeline(
        [
            LazyPerItem(),
            mark,
            StreamItems(),
            Take(2),
        ]
    )

    assert pipeline([1, 2, 3, 4]) == [10, 20]
    assert seen == [1, 2]


def test_lazy_per_item_returns_measured_iterable_when_it_is_terminal() -> None:
    pipeline = Pipeline(
        [
            LazyPerItem(),
            _multiply_by_ten,
            StreamItems(),
        ]
    )

    result = pipeline([1, 2, 3])

    assert not isinstance(result, list)
    assert iter(result) is result
    assert list(result) == [10, 20, 30]


def test_inspect_does_not_materialize_terminal_lazy_stream() -> None:
    seen: list[int] = []

    def mark(value: int) -> int:
        seen.append(value)
        return value * 10

    pipeline = Pipeline(
        [
            LazyPerItem(),
            mark,
            StreamItems(),
        ]
    )

    result = pipeline.inspect([1, 2, 3])

    assert seen == []
    assert [span.label for span in result.spans] == ["0:LazyPerItem"]
    assert result.spans[0].duration_s == 0.0
    assert result.spans[0].attributes == {}
    assert result.spans[0].child_trace is None


def test_lazy_per_item_trace_resolves_when_downstream_take_closes_stream() -> None:
    pipeline = Pipeline(
        [
            LazyPerItem(),
            _multiply_by_ten,
            StreamItems(),
            Take(2),
        ]
    )

    result = pipeline.inspect([1, 2, 3, 4])

    assert [span.label for span in result.spans] == ["0:LazyPerItem", "3:Take"]
    assert result.spans[0].attributes == {
        "seen": 2,
        "emitted": 2,
        "dropped": 0,
        "closed_early": True,
    }
    assert result.spans[0].output_value is None
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == ["1:_multiply_by_ten"]
    assert result.spans[0].child_trace.spans[0].attributes == {
        "dropped": 0,
    }


def test_per_item_trace_reports_dropped_counts_per_operator() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _short_circuit_on_two,
            _multiply_by_ten,
            CollectItems(),
        ]
    )

    result = pipeline.inspect([1, 2, 3])

    assert result.spans[0].attributes == {
        "seen": 3,
        "emitted": 2,
        "dropped": 1,
    }
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == ["1:_short_circuit_on_two", "2:_multiply_by_ten"]
    assert result.spans[0].child_trace.spans[0].attributes == {
        "dropped": 1,
    }
    assert result.spans[0].child_trace.spans[1].attributes == {
        "dropped": 0,
    }


def test_per_item_inspect_preserves_failing_child_trace() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            _raise_on_two,
            _multiply_by_ten,
            CollectItems(),
        ]
    )

    result = pipeline.inspect([1, 2, 3])

    assert [span.label for span in result.spans] == ["0:PerItem"]
    assert result.spans[0].error
    assert result.spans[0].attributes == {
        "seen": 2,
        "emitted": 1,
        "dropped": 0,
    }
    assert result.spans[0].child_trace is not None
    assert any(span.label == "1:_raise_on_two" and span.error for span in result.spans[0].child_trace.spans)


def test_lazy_per_item_trace_reports_dropped_counts_per_operator() -> None:
    pipeline = Pipeline(
        [
            LazyPerItem(),
            _short_circuit_on_two,
            _multiply_by_ten,
            StreamItems(),
            Take(2),
        ]
    )

    result = pipeline.inspect([1, 2, 3])

    assert result.spans[0].attributes == {
        "seen": 3,
        "emitted": 2,
        "dropped": 1,
        "closed_early": True,
    }
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == ["1:_short_circuit_on_two", "2:_multiply_by_ten"]
    assert result.spans[0].child_trace.spans[0].attributes == {
        "dropped": 1,
    }
    assert result.spans[0].child_trace.spans[1].attributes == {
        "dropped": 0,
    }


def test_lazy_per_item_validates_as_sequence_for_downstream_collection_ops() -> None:
    contract = Pipeline(
        [
            LazyPerItem(),
            Map(_text_length),
            StreamItems(),
            Take(2),
            AcceptIntList(),
        ]
    ).validate(
        pipeline_input_type=list[str],
        strict=True,
    )

    assert contract.input_type == list[str]
    assert contract.output_type == str


def test_per_item_pipeline_composes_primitives_for_dataset_processing() -> None:
    pipeline = Pipeline(
        [
            PerItem(),
            WrapMappingInObject(target="payload", state_factory=_make_carrier),
            MapValue(str.strip, source="payload.msg", target="cleaned"),
            FilterNotNull(source="cleaned"),
            Filter(lambda text: len(text) >= 5, source="cleaned"),
            CollectItems(),
            Distinct(source="cleaned"),
            Take(1),
        ]
    )

    result = pipeline(
        [
            _message("spam", "  hello  "),
            _message("ham", "  hello  "),
            _message("ham", "  world  "),
        ]
    )

    assert len(result) == 1
    assert isinstance(result[0], Carrier)
    assert result[0].cleaned == "hello"


def test_map_validation_threads_annotated_mapper_output() -> None:
    contract = Pipeline([Map(_text_length), AcceptInt()]).validate(
        pipeline_input_type=str,
        strict=True,
    )

    assert contract.input_type == str
    assert contract.output_type == str


def test_wrap_mapping_in_object_validation_resolves_factory_output() -> None:
    contract = Pipeline(
        [
            WrapMappingInObject(target="payload", state_factory=_make_carrier),
            AcceptCarrier(),
        ]
    ).validate(
        pipeline_input_type=dict[str, object],
        strict=True,
    )

    assert contract.input_type == dict[str, object]
    assert contract.output_type == str


def test_region_and_sequence_ops_preserve_item_type_for_validation() -> None:
    contract = Pipeline(
        [
            PerItem(),
            WrapMappingInObject(target="payload", state_factory=_make_carrier),
            MapValue(str.strip, source="payload.msg", target="cleaned"),
            FilterNotNull(source="cleaned"),
            Filter(_has_min_length_or_none, source="cleaned"),
            CollectItems(),
            Distinct(source="cleaned"),
            Take(1),
            AcceptCarrierList(),
        ]
    ).validate(
        pipeline_input_type=list[dict[str, object]],
        strict=True,
    )

    assert contract.input_type == list[dict[str, object]]
    assert contract.output_type == int
