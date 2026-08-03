from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes.tensor import TensorPayload
from ml_pipes.torch import ToDevice, ToNumpy, ToNumpyRegistry, ToTorch, ToTorchRegistry, TorchSynchronizeTensors
from ml_pipes.torch.boundary_ops import _numpy_conversion_can_alias_torch_source, _torch_conversion_can_alias_numpy_source
from ml_pipes.torch.types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TorchTensorPayload:
    return TorchTensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def test_to_torch_copy_false_shares_cpu_numpy_storage():
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(copy=False)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [9.0, 2.0]


def test_to_torch_copy_true_isolates_cpu_numpy_storage():
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(copy=True)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [1.0, 2.0]


def test_to_numpy_copy_false_shares_cpu_torch_storage():
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32), layout="N")

    result = ToNumpy(copy=False)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [9.0, 2.0]


def test_to_numpy_copy_true_isolates_cpu_torch_storage():
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32), layout="N")

    result = ToNumpy(copy=True)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [1.0, 2.0]


def test_to_torch_registry_copy_true_isolates_cpu_numpy_storage():
    from ml_pipes.tensor import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})

    result = ToTorchRegistry(copy=True)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [1.0, 2.0]


def test_to_numpy_registry_copy_false_shares_cpu_torch_storage():
    registry = TorchTensorRegistry({"scores": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    result = ToNumpyRegistry(copy=False)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [9.0, 2.0]


def test_to_torch_registry_copy_false_shares_cpu_numpy_storage():
    from ml_pipes.tensor import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})

    result = ToTorchRegistry(copy=False)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [9.0, 2.0]


def test_to_numpy_registry_copy_true_isolates_cpu_torch_storage():
    registry = TorchTensorRegistry({"scores": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    result = ToNumpyRegistry(copy=True)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [1.0, 2.0]


def test_to_device_updates_payload_and_registry_devices():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))
    moved_payload = ToDevice("cpu")(payload)
    assert moved_payload.device == "cpu"

    registry = TorchTensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})
    moved_registry = ToDevice("cpu")(registry)
    assert moved_registry["scores"].device.type == "cpu"


def test_to_device_supports_tensor_sequences_and_runtime_outputs():
    tensor = torch.ones((2,), dtype=torch.float32)
    moved_tensor = ToDevice("cpu")(tensor)
    assert moved_tensor.device.type == "cpu"

    payloads = [_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")]
    moved_payloads = ToDevice("cpu")(payloads)
    assert moved_payloads[0].device == "cpu"

    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC"),),
        names=("scores",),
    )
    moved_outputs = ToDevice("cpu")(outputs)
    assert moved_outputs.tensors[0].device == "cpu"


def test_torch_synchronize_tensors_passthrough_on_payload():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))

    result = TorchSynchronizeTensors()(payload)

    assert result is payload


def test_torch_synchronize_tensors_collects_devices_from_runtime_outputs(monkeypatch):
    outputs = TorchRuntimeOutputs(
        tensors=(
            _torch_payload(torch.ones((1, 2), dtype=torch.float32)),
            _torch_payload(torch.ones((1, 3), dtype=torch.float32)),
        ),
        names=("a", "b"),
    )
    seen: list[str] = []

    monkeypatch.setattr(
        "ml_pipes.torch.boundary_ops._synchronize_torch_device",
        lambda device: seen.append(str(device)),
    )

    result = TorchSynchronizeTensors()(outputs)

    assert result is outputs
    assert seen == ["cpu"]


def test_torch_synchronize_tensors_rejects_non_torch_values():
    with pytest.raises(TypeError, match="TorchSynchronizeTensors"):
        TorchSynchronizeTensors()(123)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_smoke_for_to_torch_and_to_device():
    payload = TensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    to_cuda = ToTorch(device="cuda:0")
    result = ToDevice("cuda:0")(to_cuda(payload))

    assert result.device == "cuda:0"
    assert result.array.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_numpy_copy_true_from_cuda_does_not_need_alias_breaking_copy():
    payload = _torch_payload(torch.ones((2,), dtype=torch.float32, device="cuda:0"), layout="N")

    assert not _numpy_conversion_can_alias_torch_source(
        source_device_type=payload.array.device.type,
        source_dtype=np.dtype("float32"),
        target_dtype=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_torch_copy_true_to_cuda_does_not_need_alias_breaking_copy():
    array = np.ones((2,), dtype=np.float32)
    source_dtype = torch.as_tensor(array).dtype

    assert not _torch_conversion_can_alias_numpy_source(
        device="cuda:0",
        source_dtype=source_dtype,
        target_dtype=None,
    )
