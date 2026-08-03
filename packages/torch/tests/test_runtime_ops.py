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


def test_torch_distribute_clones_per_sample_outputs():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((2, 4))),),
        names=("preds",),
    )

    result = TorchDistribute()(outputs)
    result[0].tensors[0].array[:] = 99

    assert torch.all(result[1].tensors[0].array == 1)
    assert result[0].tensors[0].array.data_ptr() != result[1].tensors[0].array.data_ptr()
