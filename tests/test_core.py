import pytest

from ml_pipes import (
    Context,
    DecodeOp,
    NormalizeOp,
    Pipeline,
    PipelineValidationError,
    ResizeOp,
    Value,
)


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
    value = Value(data="image")

    assert value.context.transforms == ()
    assert value.context.metadata == {}


def test_pipeline_validate_accepts_compatible_operator_chain():
    pipeline = Pipeline([DecodeOp(), ResizeOp(), NormalizeOp()])

    pipeline.validate()


def test_pipeline_validate_rejects_incompatible_operator_chain():
    pipeline = Pipeline([NormalizeOp(), ResizeOp()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_can_validate_during_initialization():
    Pipeline([DecodeOp(), ResizeOp(), NormalizeOp()], validate_on_init=True)


def test_pipeline_validate_requires_operator_annotations():
    class UntypedOp:
        def __call__(self, value):
            return value

    pipeline = Pipeline([UntypedOp()])

    with pytest.raises(PipelineValidationError, match="missing a type annotation"):
        pipeline.validate()


def test_pipeline_validate_allows_broader_downstream_input_type():
    class ProduceString:
        def __call__(self, value: int) -> str:
            return str(value)

    class ConsumeObject:
        def __call__(self, value: object) -> object:
            return value

    pipeline = Pipeline([ProduceString(), ConsumeObject()])

    pipeline.validate()
