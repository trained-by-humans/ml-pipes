"""
Tests for Store/Recall reachability validation — richer error messages,
extend() re-validation, and Embed attribution.
"""
from typing import Any

import numpy as np
import pytest

from ml_pipes.core import (
    Pipeline,
    embed,
)
from ml_pipes.standard import (
    Batch,
    Pick,
    Recall,
    Store,
    UnBatch,
)
from ml_pipes.validation import PipelineValidationError
from ml_pipes._typing.annotation import expand_annotation_parts
from ml_pipes.context import Recall, Store
from ml_pipes.vision import ImagePayload


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


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


class StringPairConsumer:
    def __call__(self, left: str, right: str) -> str:
        return f"{left}|{right}"


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


def test_store_pick_recall_type_flow_passes():
    Pipeline([
        IntToPair(),
        Store("saved_text", source=1),
        Pick(0),
        IntToString(),
        Recall("saved_text"),
        StringPairConsumer(),
    ]).validate()


def test_store_select_stores_image_payload_property():
    image = ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8))
    pipeline = Pipeline([Store("image_shape", source="spatial_shape"), Recall("image_shape")])

    result = pipeline(image)

    assert result[0] is image
    assert result[1] == (10, 20)


def test_store_select_type_flow_passes():
    Pipeline([Store("image_shape", source="spatial_shape"), Recall("image_shape")]).validate()


def test_recall_resolve_contract_keeps_variadic_tuple_item_type_when_store_matches_it():
    input_types, output_type = Recall("x").resolve_contract(
        tuple[int, ...],
        {"x": int},
        expand_annotation_parts,
        PipelineValidationError,
    )

    assert input_types == (Any,)
    assert output_type == tuple[int, ...]


def test_recall_resolve_contract_widens_variadic_tuple_item_type_for_inserted_value():
    input_types, output_type = Recall("x").resolve_contract(
        tuple[int, ...],
        {"x": tuple[int, ...]},
        expand_annotation_parts,
        PipelineValidationError,
    )

    assert input_types == (Any,)
    assert output_type == tuple[int | tuple[int, ...], ...]


def test_store_index_out_of_bounds_raises_on_concrete_input():
    with pytest.raises(PipelineValidationError, match=r"Store\('x', \(5,\)\) is out of bounds"):
        Pipeline([IntToPair(), Store("x", source=5)]).validate()


def test_store_index_out_of_bounds_silent_on_vague_input():
    Pipeline([Store("x", source=5)]).validate()


def test_store_rejects_removed_index_keyword():
    with pytest.raises(TypeError, match="unexpected keyword"):
        Store("x", index=0)


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
    with pytest.raises(PipelineValidationError, match="Pipeline step 0:Recall"):
        Pipeline([Recall("x")]).validate()


def test_error_message_contains_operator_index_two():
    with pytest.raises(PipelineValidationError, match="Pipeline step 2:Recall"):
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


def test_recall_after_other_ops_before_store_raises():
    with pytest.raises(PipelineValidationError, match="was not stored"):
        Pipeline([IntToString(), Recall("missing_value"), StringToFloat()]).validate()


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
    with pytest.raises(PipelineValidationError, match=r"inside Embed"):
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
