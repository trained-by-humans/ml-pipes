import pytest

from ml_pipes import Pipeline, PipelineValidationError, Embed, embed, Store, Recall


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

def test_embed_runs_inner_pipeline():
    inner = Pipeline([IntToString(), StringToFloat()])
    outer = Pipeline([embed(inner), FloatToInt()])

    assert outer(42) == 42


def test_embed_composes_multiple_inner_pipelines():
    to_str = Pipeline([IntToString()])
    to_float = Pipeline([StringToFloat()])
    to_int = Pipeline([FloatToInt()])

    outer = Pipeline([embed(to_str), embed(to_float), embed(to_int)])

    assert outer(7) == 7


def test_embed_result_is_passed_to_next_operator():
    inner = Pipeline([Doubled()])
    outer = Pipeline([embed(inner), IntToString()])

    assert outer(5) == "10"


# --- context isolation ---

def test_embed_does_not_expose_inner_context_to_outer():
    inner = Pipeline([IntToString(), Store("inner_val")])

    captured_context = {}

    from ml_pipes.context import ContextOp
    from typing import Any

    class CaptureContextOp(ContextOp):
        def apply(self, current, context):
            captured_context.update(context.values)
            return current, context

        def resolve_contract(self, current_output, stored_annotations, expand, err):
            return (Any,), Any

    outer = Pipeline([embed(inner), CaptureContextOp()])
    outer(3)

    assert "inner_val" not in captured_context


def test_embed_does_not_inherit_outer_context():
    outer_store = Store("outer_val")
    inner = Pipeline([Recall("outer_val")])

    outer = Pipeline([outer_store, embed(inner)])

    with pytest.raises(KeyError, match="outer_val"):
        outer(42)


def test_embed_outer_context_is_unchanged_after_inner_runs():
    class PassThrough:
        def __call__(self, value: int) -> int:
            return value

    inner = Pipeline([PassThrough()])

    stored = {}

    from ml_pipes.context import ContextOp
    from typing import Any

    class CheckContext(ContextOp):
        def apply(self, current, context):
            stored["before"] = dict(context.values)
            return current, context

        def resolve_contract(self, current_output, stored_annotations, expand, err):
            return (Any,), Any

    class CheckContextAfter(ContextOp):
        def apply(self, current, context):
            stored["after"] = dict(context.values)
            return current, context

        def resolve_contract(self, current_output, stored_annotations, expand, err):
            return (Any,), Any

    outer = Pipeline([Store("x"), CheckContext(), embed(inner), CheckContextAfter()])
    outer(1)

    assert stored["before"] == stored["after"] == {"x": 1}


# --- validator ---

def test_embed_validate_propagates_inner_output_type():
    inner = Pipeline([IntToString()])
    outer = Pipeline([embed(inner), StringToFloat()], validate_on_init=True)

    outer.validate()


def test_embed_validate_catches_type_mismatch_across_boundary():
    inner = Pipeline([IntToString()])
    outer = Pipeline([embed(inner), FloatToInt()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        outer.validate()


def test_embed_validate_chains_multiple_inner_pipelines():
    a = Pipeline([IntToString()])
    b = Pipeline([StringToFloat()])

    outer = Pipeline([embed(a), embed(b), FloatToInt()], validate_on_init=True)

    outer.validate()


def test_embed_validate_rejects_inner_recall_of_outer_store():
    inner = Pipeline([Recall("x")])
    outer = Pipeline([Store("x"), embed(inner)])

    with pytest.raises(PipelineValidationError, match="was not stored"):
        outer.validate()


def test_embed_class_and_function_are_equivalent():
    inner = Pipeline([IntToString()])

    assert isinstance(embed(inner), Embed)
    assert isinstance(Embed(inner), Embed)
