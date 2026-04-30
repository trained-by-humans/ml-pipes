import pytest
from typing import Any

from ml_pipes import Pipeline, PipelineValidationError, Scatter, Gather, Batch, UnBatch


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


class ListIntToStr:
    def __call__(self, value: list[int]) -> str:
        return "list of ints"


class ObjectConsumer:
    def __call__(self, value: object) -> object:
        return value


class ObjectProducer:
    def __call__(self, value: int) -> object:
        return value


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


class PairToBool:
    def __call__(self, number: int, text: str) -> bool:
        return text == str(number)


class TripleConsumer:
    def __call__(self, x: int, y: str, z: float) -> str:
        return f"{x}-{y}-{z}"


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value

class VagueListOp:
    def __call__(self, value: list[Any]) -> list[Any]:
        return value


class VagueToListOp:
    def __call__(self, value: Any) -> list[Any]:
        return value


class VagueListToSingleOp:
    def __call__(self, value: list[Any]) -> Any:
        return value


class StringConsumer:
    def __call__(self, value: str) -> str:
        return value


class ContractPassthrough:
    def __call__(self, value: Any) -> Any:
        return value

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, validation_error_type):
        return (Any,), current_output


def test_validate_accepts_empty_pipeline():
    Pipeline([]).validate()  # must not raise


def test_pipeline_validate_accepts_compatible_operator_chain():
    pipeline = Pipeline([IntToString(), StringToFloat(), FloatToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_incompatible_operator_chain():
    pipeline = Pipeline([IntToString(), BoolToBytes()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_auto_validate_raises_during_initialization():
    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        Pipeline([IntToString(), BoolToBytes()], auto_validate=True)


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


def test_pipeline_validate_rejects_tighter_downstream_input_type():
    pipeline = Pipeline([ObjectProducer(), StringConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_validate_accepts_tuple_output_for_multi_arg_operator():
    pipeline = Pipeline([IntToPair(), PairToBool()])

    pipeline.validate()


def test_pipeline_validate_rejects_tuple_output_with_wrong_arity():
    pipeline = Pipeline([IntToPair(), TripleConsumer()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_validate_vague_op_between_typed_ops_is_accepted():
    pipeline = Pipeline([IntToString(), VagueOp(), BoolToBytes()])

    pipeline.validate()  # must not raise


def test_resolved_input_type_uses_operator_boundary():
    contract = Pipeline([IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int
    assert contract.output_type is str


def test_resolved_input_type_defaults_to_any_when_no_constraint_is_known():
    contract = Pipeline([ContractPassthrough()]).validate()

    assert contract is not None
    assert contract.input_type is Any


def test_resolved_input_type_skips_contract_passthrough():
    inner = Pipeline([ContractPassthrough(), IntToString()])
    contract = inner.validate(inference=True)

    assert contract is not None
    assert contract.input_type is int


def test_resolved_input_type_preserves_strongest_known_boundary_through_vague_ops():
    contract = Pipeline([ContractPassthrough(), VagueListOp(), ListIntToStr()]).validate(inference=True)

    assert contract is not None
    assert contract.input_type == list[Any]


def test_resolved_input_type_backpropagates_through_contract_passthrough():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        IntToString(),
        Gather()
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[int]


def test_resolved_input_type_stops_backpropagating_once_the_boundary_is_concrete():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        IntToString(),
        StringToFloat(),
        Gather()
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[int]


def test_resolved_input_type_stops_backpropagating_at_vague_operator():
    inner = Pipeline([
        ContractPassthrough(),
        Scatter(),
        VagueOp(),
        Batch(size=2),
        ListIntToStr(),
        UnBatch(),
        Gather(),
    ])
    contract = inner.validate(inference=True)

    assert contract.input_type == list[Any]


def test_validate_does_not_run_backward_inference_by_default():
    contract = Pipeline([ContractPassthrough(), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is Any


def test_validate_prefers_more_concrete_explicit_pipeline_input_type():
    contract = Pipeline([IntToString()]).validate(pipeline_input_type=bool)

    assert contract is not None
    assert contract.input_type is bool
