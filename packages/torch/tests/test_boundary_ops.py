from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes.tensor import TensorPayload
from ml_pipes.torch import ToDevice, ToNumpy, ToNumpyRegistry, ToTorch, ToTorchRegistry, SynchronizeTensors
from ml_pipes.torch.types import RuntimeOutputs, TensorPayload, TensorRegistry


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TensorPayload:
    return TensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def _non_cpu_device_params() -> list[object]:
    params: list[object] = []
    if torch.cuda.is_available():
        params.append(pytest.param("cuda:0", id="cuda"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        params.append(pytest.param("mps", id="mps"))
    if params:
        return params
    return [pytest.param(None, marks=pytest.mark.skip(reason="No non-CPU torch device is available"), id="no-non-cpu-device")]


NON_CPU_DEVICE_PARAMS = _non_cpu_device_params()
REQUESTED_DEVICE_PARAMS = [
    pytest.param("cpu", id="cpu"),
    *NON_CPU_DEVICE_PARAMS,
]
COPY_FLAGS = [
    pytest.param(False, id="copy-false"),
    pytest.param(True, id="copy-true"),
]


@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_torch_cpu_conversion_respects_copy_flag(copy: bool):
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(copy=copy)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == ([1.0, 2.0] if copy else [9.0, 2.0])


@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_numpy_cpu_conversion_respects_copy_flag(copy: bool):
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32), layout="N")

    result = ToNumpy(copy=copy)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == ([1.0, 2.0] if copy else [9.0, 2.0])


@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_torch_registry_cpu_conversion_respects_copy_flag(copy: bool):
    from ml_pipes.tensor import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})

    result = ToTorchRegistry(copy=copy)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == ([1.0, 2.0] if copy else [9.0, 2.0])


@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_numpy_registry_cpu_conversion_respects_copy_flag(copy: bool):
    registry = TensorRegistry({"scores": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    result = ToNumpyRegistry(copy=copy)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == ([1.0, 2.0] if copy else [9.0, 2.0])


@pytest.mark.parametrize("device", NON_CPU_DEVICE_PARAMS)
@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_torch_cross_device_conversion_detaches_from_numpy_source(device: str, copy: bool):
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(device=device, copy=copy)(payload)
    payload.array[0] = 9.0

    assert result.array.cpu().tolist() == [1.0, 2.0]


@pytest.mark.parametrize("device", NON_CPU_DEVICE_PARAMS)
@pytest.mark.parametrize("copy", COPY_FLAGS)
def test_to_numpy_cross_device_conversion_detaches_from_torch_source(device: str, copy: bool):
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32, device=device), layout="N")

    result = ToNumpy(copy=copy)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [1.0, 2.0]


@pytest.mark.parametrize("device", REQUESTED_DEVICE_PARAMS)
def test_to_device_moves_payload_and_registry_to_requested_device(device: str):
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))
    moved_payload = ToDevice(device)(payload)
    assert moved_payload.device == str(torch.device(device))
    assert moved_payload.array.device.type == torch.device(device).type

    registry = TensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})
    moved_registry = ToDevice(device)(registry)
    assert moved_registry["scores"].device.type == torch.device(device).type


def test_to_device_supports_tensor_sequences_and_runtime_outputs():
    tensor = torch.ones((2,), dtype=torch.float32)
    moved_tensor = ToDevice("cpu")(tensor)
    assert moved_tensor.device.type == "cpu"

    payloads = [_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")]
    moved_payloads = ToDevice("cpu")(payloads)
    assert moved_payloads[0].device == "cpu"

    outputs = RuntimeOutputs(
        tensors=(_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC"),),
        names=("scores",),
    )
    moved_outputs = ToDevice("cpu")(outputs)
    assert moved_outputs.tensors[0].device == "cpu"


def test_torch_synchronize_tensors_passthrough_on_payload():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))

    result = SynchronizeTensors()(payload)

    assert result is payload


def test_torch_synchronize_tensors_collects_devices_from_runtime_outputs(monkeypatch):
    outputs = RuntimeOutputs(
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

    result = SynchronizeTensors()(outputs)

    assert result is outputs
    assert seen == ["cpu"]


def test_torch_synchronize_tensors_rejects_non_torch_values():
    with pytest.raises(TypeError, match="SynchronizeTensors"):
        SynchronizeTensors()(123)
