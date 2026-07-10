from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.tensor import AsType, Collate, TensorPayload, TensorRegistry
from ml_pipes.validation import PipelineValidationError


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class MakeTensor:
    def __call__(self, value: int) -> TensorPayload:
        return TensorPayload(array=np.array([value], dtype=np.float32), layout="N", dtype="float32")


class AcceptTensor:
    def __call__(self, value: TensorPayload) -> int:
        return int(value.array[0])


def test_as_type_does_not_silence_downstream_type_check():
    pipeline = Pipeline([AsType("float32"), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_as_type_establishes_tensor_like_input_contract():
    contract = Pipeline([AsType("float32")]).validate()

    assert contract is not None
    assert contract.input_type == (
        TensorPayload
        | np.ndarray
        | tuple[TensorPayload, ...]
        | tuple[np.ndarray, ...]
        | list[TensorPayload]
        | list[np.ndarray]
    )


def test_as_type_preserves_single_tensor_contract_for_typed_pipeline():
    contract = Pipeline([MakeTensor(), AsType("float16"), AcceptTensor()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_as_type_can_cast_tuple_of_tensor_payloads():
    tensors = (
        TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        TensorPayload(array=np.array([[3.0, 4.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
    )

    result = AsType("float32")(tensors)

    assert isinstance(result, tuple)
    assert result[0].array.dtype == np.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == np.float32
    assert result[1].dtype == "float32"


def test_as_type_can_cast_list_of_tensor_payloads():
    tensors = [
        TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        TensorPayload(array=np.array([[3.0, 4.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
    ]

    result = AsType("float32")(tensors)

    assert isinstance(result, list)
    assert result[0].array.dtype == np.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == np.float32
    assert result[1].dtype == "float32"


def test_as_type_can_cast_single_tensor_payload():
    payload = TensorPayload(
        array=np.array([[1.0, 2.0]], dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = AsType("float16")(payload)

    assert result.array.dtype == np.float16
    assert result.dtype == "float16"


def test_as_type_can_cast_single_array():
    array = np.array([[1.0, 2.0]], dtype=np.float16)

    result = AsType("float32")(array)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_as_type_can_cast_named_registry_tensor_in_place():
    registry = TensorRegistry({"density": np.array([[1.0, 2.0]], dtype=np.float16)})

    result = AsType(src="density", dtype="float32")(registry)

    assert result is registry
    assert result["density"].dtype == np.float32


def test_as_type_can_write_named_registry_tensor_to_new_key():
    registry = TensorRegistry({"density": np.array([[1.0, 2.0]], dtype=np.float16)})

    result = AsType(src="density", dtype="float32", as_="density_fp32")(registry)

    assert result is registry
    assert result["density"].dtype == np.float16
    assert result["density_fp32"].dtype == np.float32


def test_as_type_without_src_rejects_registry_input():
    registry = TensorRegistry({"density": np.array([[1.0, 2.0]], dtype=np.float16)})

    with pytest.raises(TypeError):
        AsType(dtype="float32")(registry)


def test_as_type_with_src_rejects_tensor_payload_input():
    payload = TensorPayload(
        array=np.array([[1.0, 2.0]], dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    with pytest.raises(TypeError):
        AsType(src="density", dtype="float32")(payload)


def test_as_type_rejects_as_without_src():
    with pytest.raises(ValueError):
        AsType(dtype="float32", as_="density_fp32")


def test_collate_concatenates_nchw_tensors_along_batch_dim():
    tensors = [
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
    ]

    result = Collate()(tensors)

    assert result.array.shape == (3, 3, 8, 8)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"


def test_collate_stacks_chw_tensors_adding_batch_dim():
    tensors = [
        TensorPayload(array=np.zeros((3, 8, 8), dtype=np.float32), layout="CHW", dtype="float32"),
        TensorPayload(array=np.zeros((3, 8, 8), dtype=np.float32), layout="CHW", dtype="float32"),
    ]

    result = Collate()(tensors)

    assert result.array.shape == (2, 3, 8, 8)


def test_collate_raises_on_empty_list():
    with pytest.raises(ValueError, match="empty"):
        Collate()([])
