from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml_pipes import (
    Distinct,
    DistinctBy,
    DropNull,
    EndForEachItem,
    EndLazyForEachItem,
    FilterNotNull,
    Filter,
    ForEachItem,
    LazyForEachItem,
    Map,
    MapNotNull,
    MapValue,
    Pipeline,
    SHORT_CIRCUIT,
    WrapMappingInObject,
    Skip,
    SkipWhile,
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


class AcceptInt:
    def __call__(self, value: int) -> str:
        return str(value)


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
class WrongPayloadCarrier:
    payload: str = ""


def _is_positive(value: int) -> bool:
    return value > 0


def _is_positive_or_none(value: int | None) -> bool:
    return value is None or value > 0


def _hash_text(text: str) -> int:
    return len(text)


def _hash_optional_text(text: str | None) -> int:
    return 0 if text is None else len(text)


def _short_circuit_on_two(value: int) -> int | object:
    if value == 2:
        return SHORT_CIRCUIT
    return value


def _multiply_by_ten(value: int) -> int:
    return value * 10


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
    result = WrapMappingInObject(target="payload", state_factory=Carrier)(_message("spam", "hello"))

    assert isinstance(result, Carrier)
    assert result.payload == {"label": "spam", "msg": "hello"}


def test_wrap_mapping_in_object_validation_returns_factory_output_when_input_is_unknown() -> None:
    contract = Pipeline([WrapMappingInObject(target="payload", state_factory=Carrier)]).validate()

    assert contract.output_type == Carrier


def test_wrap_mapping_in_object_validation_rejects_incompatible_target_annotation() -> None:
    with pytest.raises(PipelineValidationError, match="target"):
        Pipeline([WrapMappingInObject(target="payload", state_factory=WrongPayloadCarrier)]).validate(
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


def test_take_and_skip_support_iterables() -> None:
    items = (value for value in range(5))

    assert Take(2)(items) == [0, 1]
    assert Skip(2)(range(5)) == [2, 3, 4]


def test_take_closes_iterator_when_it_short_circuits() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert Take(2)(items) == [0, 1]
    assert items.closed is True


def test_skip_does_not_explicitly_close_iterator_after_full_consumption() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert Skip(2)(items) == [2, 3]
    assert items.closed is False


def test_skip_then_take_materializes_expected_slice() -> None:
    assert Take(2)(Skip(1)(range(5))) == [1, 2]


def test_take_validation_uses_static_contract() -> None:
    contract = Pipeline([Take(2), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type == str


def test_skip_validation_uses_materialized_contract() -> None:
    contract = Pipeline([Skip(2), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type == str


def test_skip_can_chain_into_take_to_materialize_again() -> None:
    contract = Pipeline([Skip(2), Take(2), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type == str


def test_take_while_and_skip_while_support_iterables() -> None:
    assert TakeWhile(lambda value: value < 3)(range(5)) == [0, 1, 2]
    assert SkipWhile(lambda value: value < 3)(range(5)) == [3, 4]


def test_take_while_closes_iterator_when_it_short_circuits() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert TakeWhile(lambda value: value < 2)(items) == [0, 1]
    assert items.closed is True


def test_skip_while_does_not_explicitly_close_iterator_after_full_consumption() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert SkipWhile(lambda value: value < 2)(items) == [2, 3]
    assert items.closed is False


def test_take_while_and_skip_while_validation_check_predicate_input_type() -> None:
    Pipeline([TakeWhile(_is_positive)]).validate(
        pipeline_input_type=list[int],
        strict=True,
    )
    Pipeline([SkipWhile(_is_positive)]).validate(
        pipeline_input_type=list[int],
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([TakeWhile(_has_min_length)]).validate(
            pipeline_input_type=list[int],
            strict=True,
        )

    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([SkipWhile(_has_min_length)]).validate(
            pipeline_input_type=list[int],
            strict=True,
        )


def test_skip_inside_for_each_uses_operator_held_state() -> None:
    first_pipeline = Pipeline(
        [
            ForEachItem(),
            Skip(1),
            EndForEachItem(),
        ]
    )
    second_pipeline = Pipeline(
        [
            ForEachItem(),
            Skip(1),
            EndForEachItem(),
        ]
    )

    assert first_pipeline([1, 2, 3]) == [2, 3]
    assert second_pipeline([1, 2, 3]) == [2, 3]


def test_drop_null_inside_for_each_drops_none_items() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            DropNull(),
            EndForEachItem(),
        ]
    )

    assert pipeline([1, None, 2]) == [1, 2]


def test_take_inside_for_each_drops_items_after_limit() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            Take(2),
            EndForEachItem(),
        ]
    )

    assert pipeline([0, 1, 2, 3]) == [0, 1]


def test_for_each_drops_short_circuited_items() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            _short_circuit_on_two,
            _multiply_by_ten,
            EndForEachItem(),
        ]
    )

    assert pipeline([1, 2, 3]) == [10, 30]


def test_for_each_sequence_ops_validate_against_item_types() -> None:
    contract = Pipeline(
        [
            ForEachItem(),
            Skip(1),
            Take(2),
            EndForEachItem(),
            AcceptIntList(),
        ]
    ).validate(
        pipeline_input_type=list[int],
        strict=True,
    )

    assert contract.input_type == list[int]
    assert contract.output_type == str


def test_for_each_sequence_ops_are_visible_in_child_trace() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            Skip(1),
            Take(2),
            EndForEachItem(),
        ]
    )

    result = pipeline.inspect([0, 1, 2, 3])

    assert [span.label for span in result.spans] == ["0:ForEachItem"]
    assert result.spans[0].output_value == [1, 2]
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == [
        "1:Skip",
        "2:Take",
    ]


def test_lazy_for_each_only_processes_consumed_items() -> None:
    seen: list[int] = []

    def mark(value: int) -> int:
        seen.append(value)
        return value * 10

    pipeline = Pipeline(
        [
            LazyForEachItem(),
            mark,
            EndLazyForEachItem(),
            Take(2),
        ]
    )

    assert pipeline([1, 2, 3, 4]) == [10, 20]
    assert seen == [1, 2]


def test_lazy_for_each_returns_measured_iterable_when_it_is_terminal() -> None:
    pipeline = Pipeline(
        [
            LazyForEachItem(),
            _multiply_by_ten,
            EndLazyForEachItem(),
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
            LazyForEachItem(),
            mark,
            EndLazyForEachItem(),
        ]
    )

    result = pipeline.inspect([1, 2, 3])

    assert seen == []
    assert [span.label for span in result.spans] == ["0:LazyForEachItem"]
    assert result.spans[0].duration_s == 0.0
    assert result.spans[0].attributes == {}
    assert result.spans[0].child_trace is None


def test_lazy_for_each_trace_resolves_when_downstream_take_closes_stream() -> None:
    pipeline = Pipeline(
        [
            LazyForEachItem(),
            _multiply_by_ten,
            EndLazyForEachItem(),
            Take(2),
        ]
    )

    result = pipeline.inspect([1, 2, 3, 4])

    assert [span.label for span in result.spans] == ["0:LazyForEachItem", "3:Take"]
    assert result.spans[0].attributes == {
        "seen": 2,
        "emitted": 2,
        "dropped": 0,
        "closed_early": True,
    }
    assert result.spans[0].output_value is None
    assert result.spans[0].child_trace is not None
    assert [span.label for span in result.spans[0].child_trace.spans] == ["1:_multiply_by_ten"]


def test_lazy_for_each_validates_as_sequence_for_downstream_collection_ops() -> None:
    contract = Pipeline(
        [
            LazyForEachItem(),
            Map(_text_length),
            EndLazyForEachItem(),
            Take(2),
            AcceptIntList(),
        ]
    ).validate(
        pipeline_input_type=list[str],
        strict=True,
    )

    assert contract.input_type == list[str]
    assert contract.output_type == str


def test_for_each_item_pipeline_composes_primitives_for_dataset_processing() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            WrapMappingInObject(target="payload", state_factory=Carrier),
            MapValue(str.strip, source="payload.msg", target="cleaned"),
            FilterNotNull(source="cleaned"),
            Filter(lambda text: len(text) >= 5, source="cleaned"),
            EndForEachItem(),
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
            WrapMappingInObject(target="payload", state_factory=Carrier),
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
            ForEachItem(),
            WrapMappingInObject(target="payload", state_factory=Carrier),
            MapValue(str.strip, source="payload.msg", target="cleaned"),
            FilterNotNull(source="cleaned"),
            Filter(_has_min_length_or_none, source="cleaned"),
            EndForEachItem(),
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
