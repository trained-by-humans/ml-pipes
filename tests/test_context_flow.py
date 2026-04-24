"""
Tests for Store/Recall reachability validation — richer error messages,
extend() re-validation, and Embed attribution.
"""
import pytest

from ml_pipes import Batch, Pipeline, PipelineValidationError, Recall, Store, UnBatch, embed
from ml_pipes.context import Recall, Store


# ---------------------------------------------------------------------------
# Operator stubs
# ---------------------------------------------------------------------------

class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class Identity:
    def __call__(self, value: int) -> int:
        return value


class _BatchIdentity:
    def __call__(self, values: list) -> list:
        return values


# ---------------------------------------------------------------------------
# Happy path — validate() must not raise
# ---------------------------------------------------------------------------

def test_store_then_recall_passes():
    Pipeline([Store("x"), Recall("x")]).validate()


def test_multiple_keys_pass():
    Pipeline([Store("a"), Store("b"), Recall("a"), Recall("b")]).validate()


def test_no_context_ops_passes():
    Pipeline([IntToString(), StringToFloat()]).validate()


def test_embed_with_valid_inner_store_recall_passes():
    inner = Pipeline([Store("x"), Recall("x")])
    Pipeline([embed(inner)]).validate()


def test_batch_region_store_then_recall_passes():
    Pipeline([Batch(size=2), Store("x"), _BatchIdentity(), UnBatch()]).validate()


def test_outer_key_survives_batch_region():
    # Store before batch; Recall after UnBatch — outer key must still be visible.
    Pipeline([Store("x"), Batch(size=2), _BatchIdentity(), UnBatch(), Recall("x")]).validate()


# ---------------------------------------------------------------------------
# Error detection — validate() must raise PipelineValidationError
# ---------------------------------------------------------------------------

def test_recall_without_store_raises():
    with pytest.raises(PipelineValidationError):
        Pipeline([Recall("x")]).validate()


def test_error_message_contains_key_name():
    with pytest.raises(PipelineValidationError, match="'x'"):
        Pipeline([Recall("x")]).validate()


def test_error_message_contains_operator_index_zero():
    with pytest.raises(PipelineValidationError, match="0:Recall"):
        Pipeline([Recall("x")]).validate()


def test_error_message_contains_operator_index_two():
    with pytest.raises(PipelineValidationError, match="2:Recall"):
        Pipeline([IntToString(), StringToFloat(), Recall("x")]).validate()


def test_error_message_shows_available_keys():
    with pytest.raises(PipelineValidationError, match=r"\['a', 'b'\]"):
        Pipeline([Store("a"), Store("b"), Recall("c")]).validate()


def test_error_message_shows_none_when_no_keys():
    with pytest.raises(PipelineValidationError, match=r"\(none\)"):
        Pipeline([Recall("x")]).validate()


def test_recall_before_store_raises():
    with pytest.raises(PipelineValidationError):
        Pipeline([Recall("x"), Store("x")]).validate()


def test_batch_recall_without_inner_store_raises():
    with pytest.raises(PipelineValidationError):
        Pipeline([Batch(size=2), Recall("x"), _BatchIdentity(), UnBatch()]).validate()


def test_batch_scoped_store_not_visible_outside_raises():
    # Store inside batch is discarded on UnBatch — outer Recall must fail.
    with pytest.raises(PipelineValidationError):
        Pipeline([Batch(size=2), Store("x"), _BatchIdentity(), UnBatch(), Recall("x")]).validate()


def test_outer_key_not_visible_inside_embed():
    # Outer Store("x"), then embed with inner Recall("x") — embed is isolated.
    inner = Pipeline([Recall("x")])
    with pytest.raises(PipelineValidationError):
        Pipeline([Store("x"), embed(inner)]).validate()


def test_embed_with_dangling_recall_raises_with_embed_attribution():
    inner = Pipeline([Recall("missing")])
    with pytest.raises(PipelineValidationError, match=r"inside 0:Embed"):
        Pipeline([embed(inner)]).validate()


# ---------------------------------------------------------------------------
# Type contract error messages include operator index
# ---------------------------------------------------------------------------

def test_type_mismatch_error_includes_operator_index():
    class IntOp:
        def __call__(self, value: int) -> int:
            return value

    class ExpectsFloat:
        def __call__(self, value: float) -> float:
            return value

    with pytest.raises(PipelineValidationError, match="1:ExpectsFloat"):
        Pipeline([IntOp(), ExpectsFloat()]).validate()


# ---------------------------------------------------------------------------
# auto_validate and extend() re-validation
# ---------------------------------------------------------------------------

def test_auto_validate_raises_at_construction():
    with pytest.raises(PipelineValidationError):
        Pipeline([Recall("x")], auto_validate=True)


def test_extend_revalidates_when_auto_validate_true():
    p = Pipeline([IntToString()], auto_validate=True)
    with pytest.raises(PipelineValidationError):
        p.extend([Recall("x")])


def test_extend_does_not_revalidate_when_auto_validate_false():
    p = Pipeline([IntToString()])
    p.extend([Recall("x")])  # silent — no exception


def test_extend_valid_context_flow_does_not_raise():
    p = Pipeline([Store("x")], auto_validate=True)
    p.extend([Recall("x")])  # valid — should not raise
