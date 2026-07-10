"""
Unit tests for the Inline marker and Pipeline._flatten.
Verifies construction-time expansion only — operator list shape,
immutability, operator identity.

Runtime and context-sharing behaviour are covered in test_composition_scenarios.py.
"""
import pytest

from ml_pipes.core import (
    Pipeline,
    Inline,
    inline,
)
from ml_pipes.validation import PipelineValidationError


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToInt:
    def __call__(self, value: float) -> int:
        return int(value)


# --- expansion ---

def test_inline_marker_is_gone_after_construction():
    outer = Pipeline([Inline(Pipeline([IntToString()]))])
    assert all(not isinstance(op, Inline) for op in outer.operators)


def test_inline_expands_to_correct_operator_count():
    inner = Pipeline([IntToString(), StringToFloat()])
    outer = Pipeline([Inline(inner), FloatToInt()])
    assert len(outer.operators) == 3


def test_inline_operators_are_same_objects_as_inner():
    a, b = IntToString(), StringToFloat()
    outer = Pipeline([Inline(Pipeline([a, b]))])
    assert outer.operators[0] is a
    assert outer.operators[1] is b


def test_inline_does_not_mutate_inner_pipeline():
    inner = Pipeline([IntToString()])
    _ = Pipeline([Inline(inner)])
    assert len(inner.operators) == 1


def test_multiple_inlines_all_expand():
    outer = Pipeline([Inline(Pipeline([IntToString()])), Inline(Pipeline([StringToFloat()])), FloatToInt()])
    assert len(outer.operators) == 3


def test_inline_preserves_operator_order():
    a, b, c = IntToString(), StringToFloat(), FloatToInt()
    outer = Pipeline([Inline(Pipeline([a, b])), c])
    assert outer.operators == [a, b, c]


# --- construction ---

def test_inline_function_returns_inline_instance():
    assert isinstance(inline(Pipeline([IntToString()])), Inline)


def test_inline_holds_reference_to_inner_pipeline():
    inner = Pipeline([IntToString()])
    assert Inline(inner).pipeline is inner


# --- validation ---

def test_inline_validate_treats_expanded_operators_as_flat_chain():
    outer = Pipeline([Inline(Pipeline([IntToString()])), StringToFloat()])
    outer.validate()


def test_nested_inline_fully_expands():
    a, b = IntToString(), StringToFloat()
    inner = Pipeline([Inline(Pipeline([a])), b])
    outer = Pipeline([Inline(inner)])

    assert outer.operators == [a, b]
    assert all(not isinstance(op, Inline) for op in outer.operators)


def test_inline_snapshots_operators_at_build_time():
    # Mutating the source pipeline after Inline expansion must not affect the result.
    op_a, op_b = IntToString(), StringToFloat()
    source = Pipeline([op_a])
    outer = Pipeline([Inline(source), FloatToInt()])

    source.extend([op_b])  # mutate source after composition

    assert op_b not in outer.operators


def test_inline_validate_catches_type_mismatch():
    outer = Pipeline([Inline(Pipeline([IntToString()])), FloatToInt()])
    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        outer.validate()
