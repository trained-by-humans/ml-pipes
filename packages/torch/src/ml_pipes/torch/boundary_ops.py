from __future__ import annotations

from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
import torch

from ml_pipes._typing.annotation import is_assignable
from ml_pipes.operator import Operator
from ml_pipes.tensor import TensorPayload, TensorRegistry

from .types import (
    TorchRuntimeOutputs,
    TorchTensorPayload,
    TorchTensorRegistry,
    canonical_torch_device,
    canonical_torch_dtype,
    resolve_torch_dtype,
)

__all__ = [
    "ToTorch",
    "ToNumpy",
    "ToTorchRegistry",
    "ToNumpyRegistry",
    "ToDevice",
    "TorchSynchronizeTensors",
]

TorchTensorLike: TypeAlias = (
    TorchTensorPayload
    | torch.Tensor
    | tuple[TorchTensorPayload, ...]
    | tuple[torch.Tensor, ...]
    | list[TorchTensorPayload]
    | list[torch.Tensor]
)
TorchTensorInput: TypeAlias = TorchTensorLike | TorchTensorRegistry
TorchTransferInput: TypeAlias = TorchTensorInput | TorchRuntimeOutputs
TorchTransferInputT = TypeVar("TorchTransferInputT", bound=TorchTransferInput)


def _synchronize_torch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _collect_torch_devices(value: object) -> set[torch.device]:
    if isinstance(value, TorchTensorPayload):
        return {value.array.device}
    if isinstance(value, TorchTensorRegistry):
        return {tensor.device for tensor in value._tensors.values()}
    if isinstance(value, TorchRuntimeOutputs):
        return {tensor.array.device for tensor in value.tensors}
    if isinstance(value, torch.Tensor):
        return {value.device}
    if isinstance(value, tuple | list):
        devices: set[torch.device] = set()
        for item in value:
            devices.update(_collect_torch_devices(item))
        return devices
    return set()


def _torch_conversion_can_alias_numpy_source(
    device: str,
    source_dtype: torch.dtype,
    target_dtype: torch.dtype | None,
) -> bool:
    return device == "cpu" and (target_dtype is None or source_dtype == target_dtype)


def _numpy_conversion_can_alias_torch_source(
    source_device_type: str,
    source_dtype: np.dtype,
    target_dtype: np.dtype | None,
) -> bool:
    return source_device_type == "cpu" and (target_dtype is None or source_dtype == target_dtype)


def _convert_numpy_array_to_torch(
    source: np.ndarray,
    *,
    device: str,
    target_dtype: torch.dtype | None,
    copy: bool,
) -> torch.Tensor:
    array = np.asarray(source)
    source_dtype = torch.as_tensor(array).dtype
    conversion_can_alias_source = _torch_conversion_can_alias_numpy_source(
        device=device,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
    )
    tensor = torch.as_tensor(array, dtype=target_dtype, device=device)
    detached_from_source = not conversion_can_alias_source
    if copy and not detached_from_source:
        tensor = tensor.clone()
    return tensor


def _convert_torch_tensor_to_numpy(
    source: torch.Tensor,
    *,
    target_dtype: np.dtype | None,
    copy: bool,
) -> np.ndarray:
    base_array = source.detach().cpu().numpy()
    conversion_can_alias_source = _numpy_conversion_can_alias_torch_source(
        source_device_type=source.device.type,
        source_dtype=base_array.dtype,
        target_dtype=target_dtype,
    )
    array = base_array
    if target_dtype is not None:
        array = array.astype(target_dtype, copy=False)
    detached_from_source = not conversion_can_alias_source or array is not base_array
    if copy and not detached_from_source:
        array = array.copy()
    return array


@Operator
class ToTorch:
    def __init__(self, device: str = "cpu", dtype: str | None = None, copy: bool = False):
        self.device = canonical_torch_device(device)
        self.dtype = dtype
        self.copy = copy

    def __call__(self, tensor_payload: TensorPayload) -> TorchTensorPayload:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensor = _convert_numpy_array_to_torch(
            tensor_payload.array,
            device=self.device,
            target_dtype=target_dtype,
            copy=self.copy,
        )
        return TorchTensorPayload(
            array=tensor,
            layout=tensor_payload.layout,
            dtype=canonical_torch_dtype(tensor.dtype),
            device=canonical_torch_device(tensor.device),
        )


@Operator
class ToNumpy:
    def __init__(self, dtype: str | None = None, copy: bool = False):
        self.dtype = dtype
        self.copy = copy

    def __call__(self, tensor_payload: TorchTensorPayload) -> TensorPayload:
        target_dtype = np.dtype(self.dtype) if self.dtype is not None else None
        array = _convert_torch_tensor_to_numpy(
            tensor_payload.array,
            target_dtype=target_dtype,
            copy=self.copy,
        )
        return TensorPayload(array=array, layout=tensor_payload.layout, dtype=str(array.dtype))


@Operator
class ToTorchRegistry:
    def __init__(self, device: str = "cpu", dtype: str | None = None, copy: bool = False):
        self.device = canonical_torch_device(device)
        self.dtype = dtype
        self.copy = copy

    def __call__(self, registry: TensorRegistry) -> TorchTensorRegistry:
        target_dtype = resolve_torch_dtype(self.dtype) if self.dtype is not None else None
        tensors = {}
        for name, value in registry._tensors.items():
            tensors[name] = _convert_numpy_array_to_torch(
                value,
                device=self.device,
                target_dtype=target_dtype,
                copy=self.copy,
            )
        return TorchTensorRegistry(tensors)


@Operator
class ToNumpyRegistry:
    def __init__(self, dtype: str | None = None, copy: bool = False):
        self.dtype = dtype
        self.copy = copy

    def __call__(self, registry: TorchTensorRegistry) -> TensorRegistry:
        arrays = {}
        for name, tensor in registry._tensors.items():
            target_dtype = np.dtype(self.dtype) if self.dtype is not None else None
            arrays[name] = _convert_torch_tensor_to_numpy(
                tensor,
                target_dtype=target_dtype,
                copy=self.copy,
            )
        return TensorRegistry(arrays)


@Operator
class ToDevice:
    def __init__(self, device: str):
        self.device = canonical_torch_device(device)

    def resolve_contract(self, upstream_annotation, error_type):
        if upstream_annotation is not Any and is_assignable(
            upstream_annotation,
            TorchTransferInput,
        ):
            return (upstream_annotation,), upstream_annotation
        return (TorchTransferInput,), TorchTransferInput

    def __call__(self, value: TorchTransferInputT) -> TorchTransferInputT:
        return cast(TorchTransferInputT, self._move_value(value))

    def _move_value(self, value: TorchTransferInput) -> TorchTransferInput:
        if isinstance(value, TorchTensorPayload):
            return self._move_payload(value)
        if isinstance(value, TorchTensorRegistry):
            for name, tensor in value._tensors.items():
                value[name] = tensor.to(device=self.device)
            return value
        if isinstance(value, TorchRuntimeOutputs):
            return TorchRuntimeOutputs(
                tensors=tuple(self._move_payload(tensor) for tensor in value.tensors),
                names=value.names,
            )
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)
        if isinstance(value, tuple):
            return cast(TorchTensorLike, tuple(self._move_sequence_item(item) for item in value))
        if isinstance(value, list):
            return cast(TorchTensorLike, [self._move_sequence_item(item) for item in value])
        raise TypeError(f"ToDevice does not support value type {type(value)!r}")

    def _move_sequence_item(self, value: object) -> TorchTensorPayload | torch.Tensor:
        if isinstance(value, TorchTensorPayload):
            return self._move_payload(value)
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)
        raise TypeError(f"ToDevice does not support sequence item type {type(value)!r}")

    def _move_payload(self, value: TorchTensorPayload) -> TorchTensorPayload:
        tensor = value.array.to(device=self.device)
        return TorchTensorPayload(
            array=tensor,
            layout=value.layout,
            dtype=canonical_torch_dtype(tensor.dtype),
            device=canonical_torch_device(tensor.device),
        )


@Operator
class TorchSynchronizeTensors:
    def resolve_contract(self, upstream_annotation, error_type):
        if upstream_annotation is not Any and is_assignable(
            upstream_annotation,
            TorchTransferInput,
        ):
            return (upstream_annotation,), upstream_annotation
        return (TorchTransferInput,), TorchTransferInput

    def __call__(self, value: TorchTransferInputT) -> TorchTransferInputT:
        devices = _collect_torch_devices(value)
        if not devices:
            raise TypeError(f"TorchSynchronizeTensors does not support value type {type(value)!r}")
        for device in sorted(devices, key=str):
            _synchronize_torch_device(device)
        return value
