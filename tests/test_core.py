import pytest

from ml_pipes import Context, Pipeline, PipelineValidationError, Value


class IntToString:
    def __call__(self, value: Value[int]) -> Value[str]:
        return Value(str(value.data), value.context)


class StringToFloat:
    def __call__(self, value: Value[str]) -> Value[float]:
        return Value(float(value.data), value.context)


class FloatToBool:
    def __call__(self, value: Value[float]) -> Value[bool]:
        return Value(value.data > 0, value.context)


class BoolToBytes:
    def __call__(self, value: Value[bool]) -> Value[bytes]:
        return Value(b"1" if value.data else b"0", value.context)


class ObjectConsumer:
    def __call__(self, value: Value[object]) -> Value[object]:
        return value


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
            lambda value: Value(value.data + 2, value.context),
            lambda value: Value(value.data * 3, value.context),
        ]
    )

    assert pipeline(4).data == 18


def test_value_default_context():
    value = Value(data="image")

    assert value.context.transforms == ()
    assert value.context.metadata == {}


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
