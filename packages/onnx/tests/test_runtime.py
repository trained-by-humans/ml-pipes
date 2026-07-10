import numpy as np
import pytest

from ml_pipes.onnx import Distribute, Extract, Infer, RuntimeOutputs
from ml_pipes.tensor import TensorPayload


def test_extract_creates_registry_with_named_tensors():
    array = np.array([[1.0, 2.0]], dtype=np.float32)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=array, layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    registry = Extract("output_0")(outputs)

    assert np.array_equal(registry["output_0"], array)


def test_extract_renames_tensor_with_as_():
    array = np.array([[1.0, 2.0]], dtype=np.float32)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=array, layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    registry = Extract("output_0", as_="preds")(outputs)

    assert np.array_equal(registry["preds"], array)


def test_extract_preserves_empty_named_tensors() -> None:
    outputs = RuntimeOutputs(
        tensors=(
            TensorPayload(array=np.zeros((0, 4), dtype=np.float32), layout="NC", dtype="float32"),
            TensorPayload(array=np.zeros((0,), dtype=np.float32), layout="N", dtype="float32"),
        ),
        names=("boxes", "scores"),
    )

    registry = Extract("boxes", "scores")(outputs)

    assert registry["boxes"].shape == (0, 4)
    assert registry["boxes"].dtype == np.float32
    assert registry["scores"].shape == (0,)
    assert registry["scores"].dtype == np.float32


def test_extract_raises_on_missing_output_name():
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=np.zeros((1,), dtype=np.float32), layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    with pytest.raises(KeyError, match="not found"):
        Extract("missing")(outputs)


def test_distribute_splits_batch_dim_into_per_sample_outputs():
    batched = np.arange(12, dtype=np.float32).reshape(3, 4)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=batched, layout="UNKNOWN", dtype="float32"),),
        names=("preds",),
    )
    result = Distribute()(outputs)

    assert len(result) == 3
    for i, sample in enumerate(result):
        assert sample.tensors[0].array.shape == (1, 4)
        assert np.array_equal(sample.tensors[0].array, batched[i : i + 1])
        assert sample.names == ("preds",)


def test_distribute_samples_do_not_share_memory_with_batch():
    batched = np.arange(8, dtype=np.float32).reshape(2, 4)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=batched, layout="UNKNOWN", dtype="float32"),),
        names=("preds",),
    )
    result = Distribute()(outputs)

    assert not np.shares_memory(result[0].tensors[0].array, batched)
    assert not np.shares_memory(result[1].tensors[0].array, batched)


def test_distribute_mutating_one_sample_does_not_affect_another():
    batched = np.ones((2, 4), dtype=np.float32)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=batched, layout="UNKNOWN", dtype="float32"),),
        names=("preds",),
    )
    result = Distribute()(outputs)

    result[0].tensors[0].array[:] = 99.0

    assert np.all(result[1].tensors[0].array == 1.0)


def test_distribute_preserves_empty_detection_axes_per_sample() -> None:
    outputs = RuntimeOutputs(
        tensors=(
            TensorPayload(array=np.zeros((2, 0, 4), dtype=np.float32), layout="NNC", dtype="float32"),
            TensorPayload(array=np.zeros((2, 0), dtype=np.float32), layout="NN", dtype="float32"),
        ),
        names=("boxes", "scores"),
    )

    result = Distribute()(outputs)

    assert len(result) == 2
    for sample in result:
        assert sample.names == ("boxes", "scores")
        assert sample.tensors[0].array.shape == (1, 0, 4)
        assert sample.tensors[0].array.dtype == np.float32
        assert sample.tensors[1].array.shape == (1, 0)
        assert sample.tensors[1].array.dtype == np.float32


def test_infer_op_requires_requested_model_dtype():
    class _FakeSession:
        def run(self, _output_names, _inputs):
            return [np.array([[1.0, 2.0]], dtype=np.float16)]

    infer = Infer.__new__(Infer)
    infer.session = _FakeSession()
    infer.input_name = "images"
    infer.input_layout = "NCHW"
    infer.model_dtype = np.dtype("float32")
    infer.output_layouts = ("UNKNOWN",)
    infer.output_names = ("output_0",)

    value = TensorPayload(
        array=np.zeros((1, 3, 8, 8), dtype=np.float16),
        layout="NCHW",
        dtype="float16",
    )
    with pytest.raises(ValueError, match="model dtype"):
        infer(value)
