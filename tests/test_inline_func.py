"""
Unit tests for the inline() function and >> operator.
Verifies construction-time flattening behaviour only — operator list shape,
immutability of sources, operator identity.

Runtime and context-sharing behaviour are covered in test_composition_scenarios.py.
"""
from ml_pipes import Pipeline, Inline, inline


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToInt:
    def __call__(self, value: float) -> int:
        return int(value)


# --- inline() ---

def test_inline_returns_inline_instance():
    inner = Pipeline([IntToString()])

    assert isinstance(inline(inner), Inline)


def test_inline_holds_reference_to_inner_pipeline():
    inner = Pipeline([IntToString()])

    assert inline(inner).pipeline is inner


# --- >> ---

def test_rshift_returns_new_pipeline():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    result = a >> b

    assert isinstance(result, Pipeline)


def test_rshift_merges_operator_lists():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    combined = a >> b

    assert len(combined.operators) == 2


def test_rshift_preserves_operator_identity():
    op_a, op_b = IntToString(), StringToFloat()
    combined = Pipeline([op_a]) >> Pipeline([op_b])

    assert combined.operators[0] is op_a
    assert combined.operators[1] is op_b


def test_rshift_does_not_mutate_left_pipeline():
    a = Pipeline([IntToString()])
    _ = a >> Pipeline([StringToFloat()])

    assert len(a.operators) == 1


def test_rshift_does_not_mutate_right_pipeline():
    b = Pipeline([StringToFloat()])
    _ = Pipeline([IntToString()]) >> b

    assert len(b.operators) == 1


def test_rshift_chains_three_pipelines():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])
    c = Pipeline([FloatToInt()])

    combined = a >> b >> c

    assert len(combined.operators) == 3


def test_inline_inside_pipeline_definition_equivalent_to_rshift():
    op_a, op_b = IntToString(), StringToFloat()
    a = Pipeline([op_a])
    b = Pipeline([op_b])

    via_rshift = a >> b
    via_inline = Pipeline([inline(a), inline(b)])

    assert [type(op) for op in via_rshift.operators] == [type(op) for op in via_inline.operators]
