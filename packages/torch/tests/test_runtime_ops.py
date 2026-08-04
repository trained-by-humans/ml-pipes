from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml_pipes.torch import TorchCollate, TorchDistribute, TorchExtract, TorchInfer
from ml_pipes.torch.types import TorchRuntimeOutputs, TorchTensorPayload


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TorchTensorPayload:
    return TorchTensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def test_torch_infer_accepts_sequence_outputs():
    class _Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            return [x + 1, x.sum(dim=1)]

    payload = _torch_payload(torch.ones((1, 3, 2, 2), dtype=torch.float32))

    result = TorchInfer(
        _Module().eval(),
        output_layouts=("NCHW", "NHW"),
    )(payload)

    assert result.names == ("output_0", "output_1")
    assert len(result.tensors) == 2
    assert tuple(result.tensors[0].array.shape) == (1, 3, 2, 2)
    assert tuple(result.tensors[1].array.shape) == (1, 2, 2)


def test_torch_infer_rejects_non_module_values():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        TorchInfer(lambda x: x)


def test_torch_infer_requires_requested_model_dtype():
    op = TorchInfer(torch.nn.Identity().eval(), dtype="float32")
    payload = _torch_payload(torch.ones((1, 3, 2, 2), dtype=torch.float16))

    with pytest.raises(ValueError, match="model dtype"):
        op(payload)


def test_torch_infer_supports_named_input_and_mapping_outputs():
    class _Module(torch.nn.Module):
        def forward(self, *, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "logits": pixel_values + 1,
                "masks": pixel_values.sum(dim=1),
            }

    payload = _torch_payload(torch.ones((1, 3, 2, 2), dtype=torch.float32))

    result = TorchInfer(
        _Module().eval(),
        input_name="pixel_values",
        output_layouts=("NCHW", "NHW"),
    )(payload)

    assert result.names == ("logits", "masks")
    assert tuple(result.tensors[0].array.shape) == (1, 3, 2, 2)
    assert tuple(result.tensors[1].array.shape) == (1, 2, 2)


def test_torch_infer_supports_mapping_outputs():
    class _Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "logits": x + 1,
                "masks": x.sum(dim=1),
            }

    payload = _torch_payload(torch.ones((1, 3, 2, 2), dtype=torch.float32))

    result = TorchInfer(
        _Module().eval(),
        output_layouts=("NCHW", "NHW"),
    )(payload)

    assert result.names == ("logits", "masks")
    assert tuple(result.tensors[0].array.shape) == (1, 3, 2, 2)
    assert tuple(result.tensors[1].array.shape) == (1, 2, 2)


def test_torch_infer_accepts_positional_input_name_and_input_layout():
    op = TorchInfer(torch.nn.Identity().eval(), None, "NHWC")
    payload = _torch_payload(torch.ones((1, 2, 2, 3), dtype=torch.float32), layout="NHWC")

    result = op(payload)

    assert result.names == ("output_0",)
    assert result.tensors[0].layout == "UNKNOWN"


def test_torch_infer_defaults_output_names_and_layouts():
    op = TorchInfer(torch.nn.Identity().eval())
    outputs = op(_torch_payload(torch.ones((1, 3, 4, 4))))

    assert outputs.names == ("output_0",)
    assert outputs.tensors[0].layout == "UNKNOWN"
    assert outputs.tensors[0].dtype == "float32"
    assert outputs.tensors[0].device == "cpu"


def test_torch_extract_creates_registry_with_named_tensors():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.tensor([[1.0, 2.0]], dtype=torch.float32), layout="UNKNOWN"),),
        names=("output_0",),
    )

    registry = TorchExtract("output_0")(outputs)

    assert torch.equal(registry["output_0"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_torch_extract_renames_tensor_with_as_():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.tensor([[1.0, 2.0]], dtype=torch.float32), layout="UNKNOWN"),),
        names=("output_0",),
    )

    registry = TorchExtract("output_0", as_="preds")(outputs)

    assert torch.equal(registry["preds"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_torch_extract_preserves_empty_named_tensors():
    outputs = TorchRuntimeOutputs(
        tensors=(
            _torch_payload(torch.zeros((0, 4), dtype=torch.float32), layout="NC"),
            _torch_payload(torch.zeros((0,), dtype=torch.float32), layout="N"),
        ),
        names=("boxes", "scores"),
    )

    registry = TorchExtract("boxes", "scores")(outputs)

    assert tuple(registry["boxes"].shape) == (0, 4)
    assert registry["boxes"].dtype == torch.float32
    assert tuple(registry["scores"].shape) == (0,)
    assert registry["scores"].dtype == torch.float32


def test_torch_extract_raises_for_missing_name():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((1, 3))),),
        names=("present",),
    )

    with pytest.raises(KeyError, match="missing"):
        TorchExtract("missing")(outputs)


def test_torch_collate_matches_numpy_shape_semantics():
    tensors = [
        _torch_payload(torch.zeros((1, 3, 8, 8))),
        _torch_payload(torch.zeros((1, 3, 8, 8))),
    ]

    result = TorchCollate()(tensors)

    assert result.array.shape == (2, 3, 8, 8)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"


def test_torch_collate_stacks_chw_tensors_adding_batch_dim():
    tensors = [
        _torch_payload(torch.zeros((3, 8, 8), dtype=torch.float32), layout="CHW"),
        _torch_payload(torch.zeros((3, 8, 8), dtype=torch.float32), layout="CHW"),
    ]

    result = TorchCollate()(tensors)

    assert tuple(result.array.shape) == (2, 3, 8, 8)


def test_torch_collate_raises_on_empty_list():
    with pytest.raises(ValueError, match="empty"):
        TorchCollate()([])


def test_torch_distribute_splits_batch_dim_into_per_sample_outputs():
    batched = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(batched, layout="UNKNOWN"),),
        names=("preds",),
    )

    result = TorchDistribute()(outputs)

    assert len(result) == 3
    for index, sample in enumerate(result):
        assert tuple(sample.tensors[0].array.shape) == (1, 4)
        assert torch.equal(sample.tensors[0].array, batched[index : index + 1])
        assert sample.names == ("preds",)


def test_torch_distribute_clones_per_sample_outputs():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((2, 4))),),
        names=("preds",),
    )

    result = TorchDistribute()(outputs)
    result[0].tensors[0].array[:] = 99

    assert torch.all(result[1].tensors[0].array == 1)
    assert result[0].tensors[0].array.data_ptr() != result[1].tensors[0].array.data_ptr()


def test_torch_distribute_preserves_empty_detection_axes_per_sample():
    outputs = TorchRuntimeOutputs(
        tensors=(
            _torch_payload(torch.zeros((2, 0, 4), dtype=torch.float32), layout="NNC"),
            _torch_payload(torch.zeros((2, 0), dtype=torch.float32), layout="NN"),
        ),
        names=("boxes", "scores"),
    )

    result = TorchDistribute()(outputs)

    assert len(result) == 2
    for sample in result:
        assert sample.names == ("boxes", "scores")
        assert tuple(sample.tensors[0].array.shape) == (1, 0, 4)
        assert sample.tensors[0].array.dtype == torch.float32
        assert tuple(sample.tensors[1].array.shape) == (1, 0)
        assert sample.tensors[1].array.dtype == torch.float32
