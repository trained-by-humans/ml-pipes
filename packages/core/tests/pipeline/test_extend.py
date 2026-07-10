"""
Unit tests for Pipeline.extend().
Mutates the pipeline in place and returns self.
"""
import pytest

from ml_pipes.core import (
    Pipeline,
    Inline,
)
from ml_pipes.standard import (
    Store,
    Recall,
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


class Doubled:
    def __call__(self, value: int) -> int:
        return value * 2


# --- identity ---

def test_extend_returns_same_pipeline_object():
    p = Pipeline([IntToString()])

    assert p.extend([StringToFloat()]) is p


def test_extend_mutates_operator_list_in_place():
    p = Pipeline([IntToString()])
    p.extend([StringToFloat()])

    assert len(p.operators) == 2


# --- operators ---

def test_extend_appends_operators_in_order():
    op_a, op_b, op_c = IntToString(), StringToFloat(), FloatToInt()
    p = Pipeline([op_a])
    p.extend([op_b, op_c])

    assert p.operators == [op_a, op_b, op_c]


def test_extend_with_empty_list_is_a_no_op():
    p = Pipeline([IntToString()])
    p.extend([])

    assert len(p.operators) == 1


def test_extend_multiple_times_accumulates():
    p = Pipeline([IntToString()])
    p.extend([StringToFloat()])
    p.extend([FloatToInt()])

    assert len(p.operators) == 3


# --- Inline inside extend ---

def test_extend_expands_inline_markers():
    inner = Pipeline([StringToFloat(), FloatToInt()])
    p = Pipeline([IntToString()])
    p.extend([Inline(inner)])

    assert len(p.operators) == 3
    assert all(not isinstance(op, Inline) for op in p.operators)


# --- execution ---

def test_extend_result_runs_correctly():
    p = Pipeline([IntToString()])
    p.extend([StringToFloat(), FloatToInt()])

    assert p(42) == 42


def test_extend_shares_context_with_existing_operators():
    p = Pipeline([Store("x")])
    p.extend([Recall("x")])

    result = p(7)
    assert result == (7, 7)


# --- validation ---

def test_extend_validate_accepts_compatible_new_operators():
    p = Pipeline([IntToString()])
    p.extend([StringToFloat()])

    p.validate()


def test_extend_validate_catches_type_mismatch():
    p = Pipeline([IntToString()])
    p.extend([FloatToInt()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        p.validate()
