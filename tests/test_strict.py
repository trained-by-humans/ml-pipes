import pytest

from ml_pipes import Batch, Pipeline, PipelineValidationError, Recall, SideEffectOp, Store, UnBatch
from ml_pipes.context import ContextOp, Context
from typing import Any


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value


class VagueInputOp:
    def __call__(self, value: Any) -> str:
        return str(value)


class VagueOutputOp:
    def __call__(self, value: str) -> Any:
        return value


class PassthroughOp(ContextOp):
    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        return current, context

    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        return (Any,), current_output  # accept anything, promise to return what I received


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_strict_accepts_fully_typed_pipeline():
    Pipeline([IntToString(), StringToFloat()], strict=True).validate()


def test_strict_skips_store_and_recall():
    Pipeline([IntToString(), Store("x"), Recall("x")], strict=True).validate()


def test_strict_skips_batch_unbatch():
    class ListIdentity:
        def __call__(self, values: list[int]) -> list[int]:
            return values

    Pipeline([Batch(size=2), ListIdentity(), UnBatch()], strict=True).validate()


def test_strict_accepts_passthrough_resolve_contract():
    Pipeline([IntToString(), PassthroughOp(), StringToFloat()], strict=True).validate()


def test_strict_accepts_leading_transparent_operator_before_concrete_input():
    Pipeline([Store("x"), IntToString(), StringToFloat()], strict=True).validate()


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_strict_rejects_vague_input_before_vague_output():
    # Input is checked first — the input violation fires even when output is also vague.
    with pytest.raises(PipelineValidationError, match="input type is unresolved"):
        Pipeline([VagueOp()], strict=True).validate()


def test_strict_rejects_vague_input_only():
    with pytest.raises(PipelineValidationError, match="input type is unresolved"):
        Pipeline([VagueInputOp()], strict=True).validate()


def test_strict_rejects_vague_output_only():
    with pytest.raises(PipelineValidationError, match="output type is unresolved"):
        Pipeline([IntToString(), VagueOutputOp()], strict=True).validate()


def test_strict_error_includes_operator_label():
    with pytest.raises(PipelineValidationError, match="1:VagueOutputOp"):
        Pipeline([IntToString(), VagueOutputOp()], strict=True).validate()


def test_strict_error_includes_fix_hint():
    with pytest.raises(PipelineValidationError, match="resolve_contract"):
        Pipeline([VagueOp()], strict=True).validate()


def test_strict_rejects_pipeline_with_no_concrete_input_boundary():
    with pytest.raises(PipelineValidationError, match="pipeline input type could not be inferred"):
        Pipeline([Store("x"), Recall("x")], strict=True).validate()


# ---------------------------------------------------------------------------
# Integration with auto_validate
# ---------------------------------------------------------------------------

def test_strict_with_auto_validate_raises_at_construction():
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        Pipeline([VagueOp()], strict=True, auto_validate=True)


def test_strict_with_auto_validate_raises_on_extend():
    p = Pipeline([IntToString()], strict=True, auto_validate=True)
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        p.extend([VagueOutputOp()])


def test_non_strict_accepts_vague_op():
    Pipeline([VagueOp()]).validate()  # must not raise


def test_vague_op_between_typed_ops_is_accepted():
    Pipeline([IntToString(), VagueOp(), StringToFloat()]).validate()


def test_vague_op_between_typed_ops_rejected_in_strict_mode():
    with pytest.raises(PipelineValidationError, match="Strict mode violation"):
        Pipeline([IntToString(), VagueOp(), StringToFloat()], strict=True).validate()


# ---------------------------------------------------------------------------
# Generic container types with Any args are vague
# ---------------------------------------------------------------------------

class ReturnsListAny:
    def __call__(self, value: int) -> list[Any]:
        return [value]


class ReturnsTupleWithAny:
    def __call__(self, value: int) -> tuple[int, Any]:
        return value, value


class AcceptsListAny:
    def __call__(self, value: list[Any]) -> int:
        return len(value)


class ReturnsListInt:
    def __call__(self, value: int) -> list[int]:
        return [value]


def test_strict_rejects_list_any_output():
    with pytest.raises(PipelineValidationError, match="output type is unresolved"):
        Pipeline([ReturnsListAny()], strict=True).validate()


def test_strict_rejects_tuple_with_any_output():
    with pytest.raises(PipelineValidationError, match="output type is unresolved"):
        Pipeline([ReturnsTupleWithAny()], strict=True).validate()


def test_strict_rejects_list_any_input():
    with pytest.raises(PipelineValidationError, match="input type is unresolved"):
        Pipeline([AcceptsListAny()], strict=True).validate()


def test_strict_accepts_concrete_generic():
    Pipeline([ReturnsListInt()], strict=True).validate()  # must not raise


# ---------------------------------------------------------------------------
# SideEffectOp
# ---------------------------------------------------------------------------

class RecordingEffect(SideEffectOp):
    def __init__(self):
        self.calls = []

    def effect(self, payload: Any) -> None:
        self.calls.append(payload)


def test_side_effect_op_rejects_call_override():
    with pytest.raises(TypeError, match="must not override __call__"):
        class BadEffect(SideEffectOp):
            def __call__(self, payload: Any) -> Any:
                return payload

            def effect(self, payload: Any) -> None:
                pass


def test_side_effect_op_threads_type():
    # SideEffectOp should pass strict mode and preserve the upstream type.
    op = RecordingEffect()
    Pipeline([IntToString(), op, StringToFloat()], strict=True).validate()


def test_side_effect_op_returns_input_unchanged():
    op = RecordingEffect()
    pipeline = Pipeline([IntToString(), op])
    result = pipeline(42)
    assert result == "42"
    assert op.calls == ["42"]


def test_save_image_passes_strict_validation(tmp_path):
    from ml_pipes import SaveImage
    from ml_pipes.types import ImagePayload
    import numpy as np

    class MakeImage:
        def __call__(self, value: int) -> ImagePayload:
            return ImagePayload(array=np.zeros((10, 10, 3), dtype=np.uint8), color_space="BGR", layout="HWC")

    op = SaveImage(tmp_path / "out.jpg")
    Pipeline([MakeImage(), op], strict=True).validate()


def test_log_detections_passes_strict_validation(tmp_path):
    import io
    from ml_pipes import LogDetections

    class MakeDetections:
        def __call__(self, value: int) -> list:
            return []

    op = LogDetections(
        model_path="model.onnx",
        image_path="img.jpg",
        annotated_image_path="ann.jpg",
        stream=io.StringIO(),
    )
    Pipeline([MakeDetections(), op], strict=True).validate()
