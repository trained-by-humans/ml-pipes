import pytest

from ml_pipes import Pipeline, PipelineValidationError, Store, Recall, inline


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


# --- execution ---

def test_rshift_produces_flat_pipeline():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    combined = a >> b

    assert isinstance(combined, Pipeline)
    assert len(combined.operators) == 2


def test_rshift_runs_operators_in_order():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat(), FloatToInt()])

    combined = a >> b

    assert combined(42) == 42


def test_inline_is_equivalent_to_rshift():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    via_rshift = a >> b
    via_inline = inline(a, b)

    assert [type(op) for op in via_rshift.operators] == [type(op) for op in via_inline.operators]


def test_rshift_does_not_mutate_source_pipelines():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    _ = a >> b

    assert len(a.operators) == 1
    assert len(b.operators) == 1


def test_rshift_chains_three_pipelines():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])
    c = Pipeline([FloatToInt()])

    combined = a >> b >> c

    assert combined(7) == 7


# --- shared context ---

def test_inline_shares_context_across_boundary():
    a = Pipeline([Store("x")])
    b = Pipeline([Recall("x")])

    combined = a >> b

    # Store in a, Recall in b — shared context means this should work at runtime
    result = combined(99)
    assert result == (99, 99)


def test_inline_validate_accepts_store_in_a_recall_in_b():
    a = Pipeline([Store("x")])
    b = Pipeline([Recall("x")])

    combined = a >> b

    combined.validate()


# --- validation ---

def test_rshift_validate_catches_type_mismatch_across_boundary():
    a = Pipeline([IntToString()])
    b = Pipeline([FloatToInt()])

    combined = a >> b

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        combined.validate()


def test_rshift_validate_accepts_compatible_chain():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    combined = a >> b

    combined.validate()
