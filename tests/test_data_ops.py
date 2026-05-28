from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from ml_pipes import (
    Distinct,
    DropNone,
    EndForEachItem,
    Filter,
    ForEachItem,
    Map,
    Pipeline,
    RequireMappingValue,
    RequireValue,
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


def _message(label: str, msg: str) -> dict[str, object]:
    return {"label": label, "msg": msg}


def _text_length(text: str) -> int:
    return len(text)


def _has_min_length(text: str) -> bool:
    return len(text) >= 5


def _int_to_text(value: int) -> str:
    return str(value)


class AcceptInt:
    def __call__(self, value: int) -> str:
        return str(value)


class AcceptOptionalInt:
    def __call__(self, value: int | None) -> str:
        return "" if value is None else str(value)


class AcceptCarrier:
    def __call__(self, value: Carrier) -> str:
        return value.cleaned or ""


class AcceptOptionalCarrier:
    def __call__(self, value: Carrier | None) -> str:
        return "" if value is None or value.cleaned is None else value.cleaned


class AcceptIntList:
    def __call__(self, items: list[int]) -> str:
        return ",".join(str(item) for item in items)


class AcceptIntIterable:
    def __call__(self, items: Iterable[int]) -> str:
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


def _is_positive(value: int) -> bool:
    return value > 0


def _hash_text(text: str) -> int:
    return len(text)


def test_map_transforms_current_value() -> None:
    assert Map(lambda value: value * 2)(3) == 6


def test_map_updates_selected_field_in_place_when_src_is_set() -> None:
    carrier = Carrier(payload={"msg": "  hello  "})

    result = Map(str.strip, src="payload.msg")(carrier)

    assert result is carrier
    assert carrier.payload == {"msg": "hello"}


def test_filter_applies_predicate_to_current_value() -> None:
    assert Filter(lambda value: value > 1)(2) == 2
    assert Filter(lambda value: value > 1)(1) is None


def test_filter_applies_predicate_to_selected_value() -> None:
    carrier = Carrier(payload={"msg": "hello"})

    assert Filter(lambda text: text == "hello", src="payload.msg")(carrier) is carrier
    assert Filter(lambda text: text == "bye", src="payload.msg")(carrier) is None


def test_filter_validation_returns_optional_output_type() -> None:
    contract = Pipeline([Filter(_is_positive)]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    assert contract.input_type == int
    assert contract.output_type == int | None


def test_filter_validation_preserves_optional_none_when_input_is_unknown() -> None:
    contract = Pipeline([Filter(_is_positive)]).validate()

    assert contract.output_type == Any | None


def test_filter_validation_requires_optional_aware_downstream_consumer() -> None:
    Pipeline([Filter(_is_positive), AcceptOptionalInt()]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([Filter(_is_positive), AcceptInt()]).validate(
            pipeline_input_type=int,
            strict=True,
        )


def test_filter_validation_checks_selected_source_type_against_predicate() -> None:
    with pytest.raises(PipelineValidationError, match="predicate expects"):
        Pipeline([Filter(_is_positive, src="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_map_validation_preserves_explicit_optional_input() -> None:
    contract = Pipeline([Map(_text_length)]).validate(
        pipeline_input_type=str | None,
        strict=True,
    )

    assert contract.input_type == str | None
    assert contract.output_type == int | None


def test_map_validation_checks_selected_source_type_against_mapper() -> None:
    with pytest.raises(PipelineValidationError, match="fn expects"):
        Pipeline([Map(_int_to_text, src="cleaned")]).validate(
            pipeline_input_type=StrictCarrier,
            strict=True,
        )


def test_require_value_drops_missing_selector() -> None:
    carrier = Carrier(payload={})

    assert RequireValue("payload.msg")(carrier) is None


def test_require_value_keeps_present_selector() -> None:
    carrier = Carrier(payload={"msg": "hello"})

    assert RequireValue("payload.msg")(carrier) is carrier


def test_require_value_validation_is_passthrough_without_selector() -> None:
    contract = Pipeline([RequireValue(), AcceptInt()]).validate(
        pipeline_input_type=int,
        strict=True,
    )

    assert contract.input_type == int
    assert contract.output_type == str


def test_require_value_validation_preserves_optional_none_when_input_is_unknown() -> None:
    contract = Pipeline([RequireValue()]).validate()

    assert contract.output_type == Any | None


def test_require_value_validation_returns_optional_with_selector() -> None:
    Pipeline([RequireValue("payload.msg"), AcceptOptionalCarrier()]).validate(
        pipeline_input_type=Carrier,
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([RequireValue("payload.msg"), AcceptCarrier()]).validate(
            pipeline_input_type=Carrier,
            strict=True,
        )


def test_filter_raises_on_missing_selector() -> None:
    carrier = Carrier(payload={})

    with pytest.raises(TypeError, match="selector"):
        Filter(lambda text: True, src="payload.msg")(carrier)


def test_require_mapping_value_seeds_state_factory() -> None:
    result = RequireMappingValue(as_="payload", state_factory=Carrier)(_message("spam", "hello"))

    assert isinstance(result, Carrier)
    assert result.payload == {"label": "spam", "msg": "hello"}


def test_require_mapping_value_validation_returns_optional_state_when_input_is_unknown() -> None:
    contract = Pipeline([RequireMappingValue(as_="payload", state_factory=Carrier)]).validate()

    assert contract.output_type == Carrier | None


def test_drop_none_removes_dropped_values() -> None:
    assert DropNone()([1, None, 2, None]) == [1, 2]


def test_drop_none_validation_uses_static_contract() -> None:
    contract = Pipeline([DropNone(), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int | None, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int | None, ...]
    assert contract.output_type == str


def test_distinct_deduplicates_by_selected_value() -> None:
    items = [
        Carrier(payload=_message("spam", "one"), cleaned="dup"),
        Carrier(payload=_message("ham", "two"), cleaned="dup"),
        Carrier(payload=_message("ham", "three"), cleaned="unique"),
    ]

    result = Distinct(src="cleaned")(items)

    assert [item.payload["msg"] for item in result if item.payload is not None] == ["one", "three"]


def test_distinct_validation_checks_key_input_type() -> None:
    Pipeline([Distinct(key=_hash_text)]).validate(
        pipeline_input_type=list[str],
        strict=True,
    )

    with pytest.raises(PipelineValidationError, match="key expects"):
        Pipeline([Distinct(key=_is_positive)]).validate(
            pipeline_input_type=list[str],
            strict=True,
        )


def test_take_and_skip_support_iterables() -> None:
    items = (value for value in range(5))

    assert Take(2)(items) == [0, 1]
    assert list(Skip(2)(range(5))) == [2, 3, 4]


def test_take_closes_iterator_when_it_short_circuits() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert Take(2)(items) == [0, 1]
    assert items.closed is True


def test_skip_does_not_explicitly_close_iterator_after_full_consumption() -> None:
    items = ClosableIterator([0, 1, 2, 3])

    assert list(Skip(2)(items)) == [2, 3]
    assert items.closed is False


def test_take_closes_upstream_iterator_through_lazy_skip() -> None:
    items = ClosableIterator([0, 1, 2, 3, 4])

    assert Take(2)(Skip(1)(items)) == [1, 2]
    assert items.closed is True


def test_take_validation_uses_static_contract() -> None:
    contract = Pipeline([Take(2), AcceptIntList()]).validate(
        pipeline_input_type=tuple[int, ...],
        strict=True,
    )

    assert contract.input_type == tuple[int, ...]
    assert contract.output_type == str


def test_skip_validation_uses_iterable_contract() -> None:
    contract = Pipeline([Skip(2), AcceptIntIterable()]).validate(
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


def test_for_each_item_pipeline_composes_primitives_for_dataset_processing() -> None:
    pipeline = Pipeline(
        [
            ForEachItem(),
            RequireMappingValue(as_="payload", state_factory=Carrier),
            Map(str.strip, src="payload.msg", as_="cleaned"),
            Filter(lambda text: len(text) >= 5, src="cleaned"),
            EndForEachItem(),
            DropNone(),
            Distinct(src="cleaned"),
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


def test_require_mapping_value_validation_resolves_factory_output() -> None:
    contract = Pipeline(
        [
            RequireMappingValue(as_="payload", state_factory=Carrier),
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
            RequireMappingValue(as_="payload", state_factory=Carrier),
            Map(str.strip, src="payload.msg", as_="cleaned"),
            Filter(_has_min_length, src="cleaned"),
            EndForEachItem(),
            DropNone(),
            Distinct(src="cleaned"),
            Take(1),
            AcceptCarrierList(),
        ]
    ).validate(
        pipeline_input_type=list[dict[str, object]],
        strict=True,
    )

    assert contract.input_type == list[dict[str, object]]
    assert contract.output_type == int
