"""
Unit tests for the inline() function and + operator.
a + b flattens two pipelines into one with shared context.

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
    assert isinstance(inline(Pipeline([IntToString()])), Inline)


def test_inline_holds_reference_to_inner_pipeline():
    inner = Pipeline([IntToString()])
    assert inline(inner).pipeline is inner


# --- + ---

def test_add_returns_new_pipeline():
    result = Pipeline([IntToString()]) + Pipeline([StringToFloat()])
    assert isinstance(result, Pipeline)


def test_add_merges_operator_lists():
    combined = Pipeline([IntToString()]) + Pipeline([StringToFloat()])
    assert len(combined.operators) == 2


def test_add_preserves_operator_identity():
    op_a, op_b = IntToString(), StringToFloat()
    combined = Pipeline([op_a]) + Pipeline([op_b])

    assert combined.operators[0] is op_a
    assert combined.operators[1] is op_b


def test_add_does_not_mutate_left_pipeline():
    a = Pipeline([IntToString()])
    _ = a + Pipeline([StringToFloat()])
    assert len(a.operators) == 1


def test_add_does_not_mutate_right_pipeline():
    b = Pipeline([StringToFloat()])
    _ = Pipeline([IntToString()]) + b
    assert len(b.operators) == 1


def test_add_chains_three_pipelines():
    combined = Pipeline([IntToString()]) + Pipeline([StringToFloat()]) + Pipeline([FloatToInt()])
    assert len(combined.operators) == 3


def test_inline_inside_definition_equivalent_to_add():
    op_a, op_b = IntToString(), StringToFloat()
    a, b = Pipeline([op_a]), Pipeline([op_b])

    via_add = a + b
    via_inline = Pipeline([inline(a), inline(b)])

    assert [type(op) for op in via_add.operators] == [type(op) for op in via_inline.operators]
