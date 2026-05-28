from __future__ import annotations

from dataclasses import dataclass

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
)


@dataclass
class Carrier:
    payload: dict[str, object] | None = None
    cleaned: str | None = None


def _message(label: str, msg: str) -> dict[str, object]:
    return {"label": label, "msg": msg}


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


def test_require_value_drops_missing_selector() -> None:
    carrier = Carrier(payload={})

    assert RequireValue("payload.msg")(carrier) is None


def test_require_value_keeps_present_selector() -> None:
    carrier = Carrier(payload={"msg": "hello"})

    assert RequireValue("payload.msg")(carrier) is carrier


def test_filter_raises_on_missing_selector() -> None:
    carrier = Carrier(payload={})

    with pytest.raises(TypeError, match="selector"):
        Filter(lambda text: True, src="payload.msg")(carrier)


def test_require_mapping_value_seeds_state_factory() -> None:
    result = RequireMappingValue(as_="payload", state_factory=Carrier)(_message("spam", "hello"))

    assert isinstance(result, Carrier)
    assert result.payload == {"label": "spam", "msg": "hello"}


def test_drop_none_removes_dropped_values() -> None:
    assert DropNone()([1, None, 2, None]) == [1, 2]


def test_distinct_deduplicates_by_selected_value() -> None:
    items = [
        Carrier(payload=_message("spam", "one"), cleaned="dup"),
        Carrier(payload=_message("ham", "two"), cleaned="dup"),
        Carrier(payload=_message("ham", "three"), cleaned="unique"),
    ]

    result = Distinct(src="cleaned")(items)

    assert [item.payload["msg"] for item in result if item.payload is not None] == ["one", "three"]


def test_take_and_skip_support_iterables() -> None:
    items = (value for value in range(5))

    assert Take(2)(items) == [0, 1]
    assert Skip(2)(range(5)) == [2, 3, 4]


def test_take_while_and_skip_while_support_iterables() -> None:
    assert TakeWhile(lambda value: value < 3)(range(5)) == [0, 1, 2]
    assert SkipWhile(lambda value: value < 3)(range(5)) == [3, 4]


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
