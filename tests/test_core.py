from collections.abc import Iterable
from typing import TypeVar

import pytest

from ml_pipes import Context, Pipeline, SHORT_CIRCUIT

_T = TypeVar("_T")


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class BoolToBytes:
    def __call__(self, value: bool) -> bytes:
        return b"1" if value else b"0"


class ReturnShortCircuit:
    def __call__(self, value: object) -> object:
        del value
        return SHORT_CIRCUIT


class FailIfCalled:
    def __init__(self) -> None:
        self.called = False

    def __call__(self, value: object) -> object:
        del value
        self.called = True
        raise AssertionError("downstream operator should not run after SHORT_CIRCUIT")


class VariadicTupleConsumer:
    def __call__(self, value: tuple[int, ...]) -> tuple[int, ...]:
        return value


class GenericVariadicTupleConsumer:
    def __call__(self, value: tuple[_T, ...]) -> tuple[_T, ...]:
        return value


class IntIterableConsumer:
    def __call__(self, value: Iterable[int]) -> tuple[int, ...]:
        return tuple(value)


class GenericIterableConsumer:
    def __call__(self, value: Iterable[_T]) -> tuple[_T, ...]:
        return tuple(value)


class VariadicCollector:
    def __call__(self, *values: object) -> tuple[object, ...]:
        return values


class MixedVariadicConsumer:
    def __call__(self, value: int, *rest: int) -> tuple[int, ...]:
        return (value, *rest)


class KeywordOnlyConsumer:
    def __call__(self, value: int, *, scale: int) -> int:
        return value * scale


class VarKeywordConsumer:
    def __call__(self, value: int, **metadata: object) -> int:
        del metadata
        return value


def test_context_add_returns_new_context():
    context = Context()
    next_context = context.store("resize_transform", "resize")

    assert context.values == {}
    assert next_context.values == {"resize_transform": "resize"}


def test_pipeline_applies_operators_in_order():
    pipeline = Pipeline(
        [
            lambda value: value + 2,
            lambda value: value * 3,
        ]
    )

    assert pipeline(4) == 18


def test_value_default_context():
    context = Context()

    assert context.values == {}


def test_pipeline_unpacks_tuple_output_into_next_operator():
    class IntToPair:
        def __call__(self, value: int) -> tuple[int, str]:
            return value, str(value)

    class PairToString:
        def __call__(self, number: int, text: str) -> str:
            return f"{number}:{text}"

    pipeline = Pipeline([IntToPair(), PairToString()])

    assert pipeline(7) == "7:7"


@pytest.mark.parametrize(
    ("operator", "parameter_name"),
    [
        pytest.param(VariadicCollector(), "values", id="variadic-only"),
        pytest.param(MixedVariadicConsumer(), "rest", id="mixed-fixed-and-variadic"),
    ],
)
def test_pipeline_rejects_variadic_positional_operator_parameters(operator, parameter_name):
    pipeline = Pipeline([operator])

    with pytest.raises(TypeError, match=rf"variadic positional parameters.*{parameter_name}"):
        pipeline(7)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(KeywordOnlyConsumer(), id="keyword-only"),
        pytest.param(VarKeywordConsumer(), id="var-keyword"),
    ],
)
def test_pipeline_rejects_other_non_positional_operator_parameters(operator):
    pipeline = Pipeline([operator])

    with pytest.raises(TypeError, match="chains operators by argument position"):
        pipeline(7)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        pytest.param(VariadicTupleConsumer(), (1, 2, 3), id="tuple[int,...]-to-tuple[int,...]"),
        pytest.param(GenericVariadicTupleConsumer(), (1, 2, 3), id="tuple[int,...]-to-tuple[T,...]"),
        pytest.param(IntIterableConsumer(), (1, 2, 3), id="tuple[int,...]-to-Iterable[int]"),
        pytest.param(GenericIterableConsumer(), (1, 2, 3), id="tuple[int,...]-to-Iterable[T]"),
    ],
)
def test_pipeline_passes_variadic_tuple_as_single_argument(operator, expected):
    pipeline = Pipeline([operator])

    assert pipeline((1, 2, 3)) == expected


def test_pipeline_can_store_select_and_recall_values():
    from ml_pipes import Pick, Recall, Store

    class IntToPair:
        def __call__(self, value: int) -> tuple[int, str]:
            return value, str(value)

    class StringPairConsumer:
        def __call__(self, left: str, right: str) -> str:
            return f"{left}|{right}"

    pipeline = Pipeline(
        [
            IntToPair(),
            Store("saved_text", source=1),
            Pick(0),
            IntToString(),
            Recall("saved_text"),
            StringPairConsumer(),
        ]
    )

    assert pipeline(9) == "9|9"


def test_pipeline_short_circuit_stops_execution() -> None:
    fail = FailIfCalled()
    pipeline = Pipeline([IntToString(), ReturnShortCircuit(), fail])

    assert pipeline(7) is SHORT_CIRCUIT
    assert fail.called is False
