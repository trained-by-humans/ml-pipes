"""
Unit tests for the Embed operator and >> operator.
a >> b feeds a into b as an isolated step.

Context isolation behaviour is covered in test_composition_scenarios.py.
"""
import pytest
from typing import Any

from ml_pipes import Pipeline, PipelineValidationError, Embed, embed


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToInt:
    def __call__(self, value: float) -> int:
        return int(value)


class Doubled:
    def __call__(self, value: int) -> int:
        return value * 2


# --- Embed.__call__ ---

def test_embed_call_returns_inner_pipeline_result():
    assert embed(Pipeline([IntToString(), StringToFloat()]))(42) == 42.0


def test_embed_call_passes_value_through_full_inner_chain():
    assert embed(Pipeline([Doubled(), Doubled()]))(3) == 12


# --- Embed.resolve_contract ---

def test_resolve_contract_returns_inner_input_and_output_types():
    op = embed(Pipeline([IntToString()]))

    input_types, output_type = op.resolve_contract(int, {}, None, PipelineValidationError)

    assert input_types == (int,)
    assert output_type is str


def test_resolve_contract_with_no_upstream_type_does_not_raise():
    op = embed(Pipeline([IntToString()]))

    _, output_type = op.resolve_contract(None, {}, None, PipelineValidationError)

    assert output_type is str


def test_resolve_contract_raises_on_incompatible_upstream_type():
    op = embed(Pipeline([IntToString()]))

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        op.resolve_contract(float, {}, None, PipelineValidationError)


def test_resolve_contract_accepts_subclass_of_expected_input():
    op = embed(Pipeline([IntToString()]))

    _, output_type = op.resolve_contract(bool, {}, None, PipelineValidationError)

    assert output_type is str


def test_resolve_contract_propagates_output_type_of_multi_op_inner():
    op = embed(Pipeline([IntToString(), StringToFloat(), FloatToInt()]))

    _, output_type = op.resolve_contract(int, {}, None, PipelineValidationError)

    assert output_type is int


# --- construction ---

def test_embed_and_Embed_produce_same_instance_type():
    inner = Pipeline([IntToString()])

    assert isinstance(embed(inner), Embed)
    assert isinstance(Embed(inner), Embed)


def test_embed_holds_reference_to_inner_pipeline():
    inner = Pipeline([IntToString()])
    assert embed(inner).pipeline is inner


# --- >> operator ---

def test_rshift_returns_new_pipeline():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])
    c = a >> b

    assert c is not a
    assert c is not b


def test_rshift_produces_two_embed_operators():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    combined = a >> b

    assert len(combined.operators) == 2
    assert isinstance(combined.operators[0], Embed)
    assert isinstance(combined.operators[1], Embed)
    assert combined.operators[0].pipeline is a
    assert combined.operators[1].pipeline is b


def test_rshift_does_not_mutate_left_pipeline():
    a = Pipeline([IntToString()])
    _ = a >> Pipeline([StringToFloat()])
    assert len(a.operators) == 1


def test_rshift_does_not_mutate_right_pipeline():
    b = Pipeline([StringToFloat()])
    _ = Pipeline([IntToString()]) >> b
    assert len(b.operators) == 1


def test_rshift_holds_live_reference_to_left_pipeline():
    extra = IntToString()
    a = Pipeline([IntToString()])
    composed = a >> Pipeline([StringToFloat()])

    a.extend([extra])  # mutate a after composition

    assert composed.operators[0].pipeline is a
    assert extra in composed.operators[0].pipeline.operators


def test_rshift_holds_live_reference_to_right_pipeline():
    extra = FloatToInt()
    b = Pipeline([StringToFloat()])
    composed = Pipeline([IntToString()]) >> b

    b.extend([extra])  # mutate b after composition

    assert composed.operators[-1].pipeline is b
    assert extra in composed.operators[-1].pipeline.operators


def test_rshift_chains_are_flat():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])
    c = Pipeline([FloatToInt()])

    combined = a >> b >> c

    # a >> b >> c == (a >> b) >> c — flat list of three Embed operators
    assert len(combined.operators) == 3
    assert all(isinstance(op, Embed) for op in combined.operators)
    assert combined.operators[0].pipeline is a
    assert combined.operators[1].pipeline is b
    assert combined.operators[2].pipeline is c
