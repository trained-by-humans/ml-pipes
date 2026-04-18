"""
Unit tests for the Embed operator in isolation.
Calls __call__ and resolve_contract directly — no outer Pipeline.

Context isolation and composed pipeline behaviour are covered in
test_composition_scenarios.py.
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


# --- __call__ ---

def test_embed_call_returns_inner_pipeline_result():
    op = embed(Pipeline([IntToString(), StringToFloat()]))

    assert op(42) == 42.0


def test_embed_call_passes_value_through_full_inner_chain():
    op = embed(Pipeline([Doubled(), Doubled()]))

    assert op(3) == 12


# --- resolve_contract ---

def test_resolve_contract_returns_inner_input_and_output_types():
    op = embed(Pipeline([IntToString()]))

    input_types, output_type = op.resolve_contract(int, {}, None, PipelineValidationError)

    assert input_types == (int,)
    assert output_type is str


def test_resolve_contract_with_no_upstream_type_does_not_raise():
    op = embed(Pipeline([IntToString()]))

    input_types, output_type = op.resolve_contract(None, {}, None, PipelineValidationError)

    assert output_type is str


def test_resolve_contract_raises_on_incompatible_upstream_type():
    op = embed(Pipeline([IntToString()]))  # expects int

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        op.resolve_contract(float, {}, None, PipelineValidationError)


def test_resolve_contract_accepts_subclass_of_expected_input():
    op = embed(Pipeline([IntToString()]))

    # bool is a subclass of int — compatible
    input_types, output_type = op.resolve_contract(bool, {}, None, PipelineValidationError)

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
