import pytest

from ml_pipes import Context, Pipeline, PipelineValidationError, Recall, Select, Store


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToBool:
    def __call__(self, value: float) -> bool:
        return value > 0


class BoolToBytes:
    def __call__(self, value: bool) -> bytes:
        return b"1" if value else b"0"


class ObjectConsumer:
    def __call__(self, value: object) -> object:
        return value


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


class PairToString:
    def __call__(self, number: int, text: str) -> str:
        return f"{number}:{text}"


class PairToBool:
    def __call__(self, number: int, text: str) -> bool:
        return text == str(number)


class TripleConsumer:
    def __call__(self, x: int, y: str, z: float) -> str:
        return f"{x}-{y}-{z}"


class StringPairConsumer:
    def __call__(self, left: str, right: str) -> str:
        return f"{left}|{right}"


def test_context_add_returns_new_context():
    context = Context()
    next_context = context.add("resize")

    assert context.transforms == ()
    assert context.metadata == {}
    assert next_context.transforms == ("resize",)
    assert next_context.metadata == {}


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

    assert context.transforms == ()
    assert context.metadata == {}


def test_pipeline_unpacks_tuple_output_into_next_operator():
    pipeline = Pipeline([IntToPair(), PairToString()])

    assert pipeline(7) == "7:7"


def test_pipeline_validate_accepts_compatible_operator_chain():
    pipeline = Pipeline([IntToString(), StringToFloat(), FloatToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_incompatible_operator_chain():
    pipeline = Pipeline([IntToString(), BoolToBytes()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_can_validate_during_initialization():
    Pipeline([IntToString(), StringToFloat(), FloatToBool()], validate_on_init=True)


def test_pipeline_validate_requires_operator_annotations():
    class UntypedOp:
        def __call__(self, value):
            return value

    pipeline = Pipeline([UntypedOp()])

    with pytest.raises(PipelineValidationError, match="missing a type annotation"):
        pipeline.validate()


def test_pipeline_validate_allows_broader_downstream_input_type():
    pipeline = Pipeline([IntToString(), ObjectConsumer()])

    pipeline.validate()


def test_pipeline_validate_accepts_tuple_output_for_multi_arg_operator():
    pipeline = Pipeline([IntToPair(), PairToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_tuple_output_with_wrong_arity():
    pipeline = Pipeline([IntToPair(), TripleConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_can_store_select_and_recall_values():
    pipeline = Pipeline(
        [
            IntToPair(),
            Store("saved_text", index=1),
            Select(0),
            IntToString(),
            Recall("saved_text"),
            StringPairConsumer(),
        ]
    )

    assert pipeline(9) == "9|9"


def test_pipeline_validate_accepts_store_select_and_recall():
    pipeline = Pipeline(
        [
            IntToPair(),
            Store("saved_text", index=1),
            Select(0),
            IntToString(),
            Recall("saved_text"),
            StringPairConsumer(),
        ]
    )

    pipeline.validate()


def test_pipeline_validate_rejects_recall_before_store():
    pipeline = Pipeline([IntToString(), Recall("missing_value"), StringPairConsumer()])

    with pytest.raises(PipelineValidationError, match="was not stored"):
        pipeline.validate()
