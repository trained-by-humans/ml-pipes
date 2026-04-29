import pytest
from typing import Any

from ml_pipes import Cast, Context, Pick, Pipeline, PipelineValidationError, Recall, Store, embed


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
    Pipeline([IntToString(), StringToFloat(), FloatToBool()], auto_validate=True)


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
            Pick(0),
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
            Pick(0),
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


def test_pipeline_validate_propagates_element_type_through_pick():
    # Pick(0) on tuple[int, str] should produce int, which IntToString accepts.
    pipeline = Pipeline([IntToPair(), Pick(0), IntToString()])

    pipeline.validate()


def test_embed_enforces_type_contract_at_boundary():
    # embed() calls _resolve_type_contract on the inner pipeline to check the
    # boundary type — a mismatch between outer output and inner input must raise.
    inner = Pipeline([StringToFloat()])
    outer = Pipeline([IntToString(), embed(inner)])

    outer.validate()  # int -> str -> float: compatible


def test_embed_rejects_incompatible_boundary_type():
    inner = Pipeline([StringToFloat()])
    outer = Pipeline([BoolToBytes(), embed(inner)])  # bytes -> str: incompatible

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        outer.validate()


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value


def test_outer_strict_rejects_embed_with_vague_output():
    inner = Pipeline([VagueOp()])
    outer = Pipeline([embed(inner)], strict=True)
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        outer.validate()


def test_outer_strict_accepts_embed_with_concrete_output():
    inner = Pipeline([IntToString()])
    outer = Pipeline([embed(inner)], strict=True)
    outer.validate()  # must not raise


def test_inner_strict_rejects_vague_op_regardless_of_outer():
    inner = Pipeline([VagueOp()], strict=True)
    outer = Pipeline([embed(inner)])
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        outer.validate()


def test_embed_validates_batch_pairs_in_inner_pipeline():
    from ml_pipes import Batch, UnBatch

    inner = Pipeline([Batch(size=2)])  # no matching UnBatch
    outer = Pipeline([IntToString(), embed(inner)])

    with pytest.raises(PipelineValidationError):
        outer.validate()


def test_embed_validates_context_interactions_in_inner_pipeline():
    from ml_pipes import Recall

    inner = Pipeline([Recall("x")])  # key never stored
    outer = Pipeline([IntToString(), embed(inner)])

    with pytest.raises(PipelineValidationError, match="was not stored"):
        outer.validate()


def test_rshift_enforces_type_contract_across_pipeline_boundary():
    # >> also uses _resolve_type_contract to validate the join boundary.
    left = Pipeline([IntToString()])
    right = Pipeline([BoolToBytes()])  # expects bool, gets str

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        (left >> right).validate()


def test_validate_raises_on_empty_pipeline_operators():
    # An empty pipeline skips validation entirely — no error.
    Pipeline([]).validate()  # must not raise


def test_validate_raises_on_type_mismatch():
    pipeline = Pipeline([IntToString(), BoolToBytes()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pipeline_validate_rejects_wrong_type_downstream_of_pick():
    # Pick(0) on tuple[int, str] produces int; StringToFloat expects str — mismatch.
    pipeline = Pipeline([IntToPair(), Pick(0), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pick_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Pick(5)])

    with pytest.raises(PipelineValidationError, match="Pick\\(5\\) is out of bounds"):
        pipeline.validate()


def test_pick_out_of_bounds_silent_on_vague_input():
    # No concrete tuple type upstream — Pick silently returns Any, no error.
    pipeline = Pipeline([Pick(5)])

    pipeline.validate()  # must not raise


def test_store_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Store("x", index=5)])

    with pytest.raises(PipelineValidationError, match="Store\\('x', index=5\\) is out of bounds"):
        pipeline.validate()


def test_store_out_of_bounds_silent_on_vague_input():
    # No concrete tuple type upstream — Store silently stores Any, no error.
    pipeline = Pipeline([Store("x", index=5)])

    pipeline.validate()  # must not raise


def test_resolved_input_type_uses_first_concrete_operator():
    inner = Pipeline([IntToString()])
    contract = inner.validate()

    assert contract.input_type is int
    assert contract.output_type is str


def test_resolved_input_type_skips_transparent_leading_ops():
    inner = Pipeline([Store("x"), IntToString()])
    contract = inner.validate()

    assert contract.input_type is int


def test_embed_rejects_when_transparent_prefix_hides_concrete_input():
    inner = Pipeline([Store("x"), IntToString()])
    outer = Pipeline([BoolToBytes(), embed(inner)])  # bytes -> int: incompatible

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        outer.validate()


# ---------------------------------------------------------------------------
# Passthrough resolve_contract must not propagate None as "unknown upstream"
# ---------------------------------------------------------------------------

def test_cast_does_not_silence_downstream_type_check():
    # Cast(float32) cannot produce str, so StringToFloat(str) must be caught.
    pipeline = Pipeline([Cast("float32"), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_cast_establishes_tensorpayload_input_contract():
    from ml_pipes.types import TensorPayload

    contract = Pipeline([Cast("float32")]).validate()

    assert contract is not None
    assert contract.input_type == (TensorPayload | tuple[TensorPayload, ...])


def test_vague_op_between_typed_ops_is_accepted():
    # Any->Any is compatible with everything — validation passes in non-strict mode.
    pipeline = Pipeline([IntToString(), VagueOp(), BoolToBytes()])

    pipeline.validate()  # must not raise


def test_vague_op_rejected_in_strict_mode():
    pipeline = Pipeline([IntToString(), VagueOp(), BoolToBytes()], strict=True)

    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        pipeline.validate()
