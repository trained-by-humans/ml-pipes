import pytest

from ml_pipes import Pipeline, PipelineValidationError, Inline, Store, Recall


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


# --- construction ---

def test_inline_expands_operators_at_construction():
    inner = Pipeline([IntToString(), StringToFloat()])
    outer = Pipeline([Inline(inner), FloatToInt()])

    assert len(outer.operators) == 3
    assert all(not isinstance(op, Inline) for op in outer.operators)


def test_inline_operators_are_same_objects_as_inner():
    a, b = IntToString(), StringToFloat()
    inner = Pipeline([a, b])
    outer = Pipeline([Inline(inner)])

    assert outer.operators[0] is a
    assert outer.operators[1] is b


def test_inline_does_not_mutate_inner_pipeline():
    inner = Pipeline([IntToString()])
    _ = Pipeline([Inline(inner), StringToFloat()])

    assert len(inner.operators) == 1


def test_multiple_inline_markers_all_expand():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])
    outer = Pipeline([Inline(a), Inline(b), FloatToInt()])

    assert len(outer.operators) == 3


# --- execution ---

def test_inline_runs_operators_in_order():
    inner = Pipeline([IntToString(), StringToFloat()])
    outer = Pipeline([Inline(inner), FloatToInt()])

    assert outer(42) == 42


def test_inline_shares_context_with_outer_pipeline():
    inner = Pipeline([Recall("x")])
    outer = Pipeline([Store("x"), Inline(inner)])

    result = outer(7)
    assert result == (7, 7)


def test_inline_store_in_inner_visible_to_outer():
    inner = Pipeline([Store("x")])
    outer = Pipeline([Inline(inner), Recall("x")])

    result = outer(5)
    assert result == (5, 5)


# --- validation ---

def test_inline_validate_treats_expanded_operators_as_flat_chain():
    inner = Pipeline([IntToString()])
    outer = Pipeline([Inline(inner), StringToFloat()])

    outer.validate()


def test_inline_validate_catches_type_mismatch():
    inner = Pipeline([IntToString()])
    outer = Pipeline([Inline(inner), FloatToInt()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        outer.validate()
