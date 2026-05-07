from __future__ import annotations

from dataclasses import dataclass

import torch


_TORCH_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}

_NAME_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    name: dtype for dtype, name in _TORCH_DTYPE_TO_NAME.items()
}


def canonical_torch_dtype(dtype: torch.dtype) -> str:
    return _TORCH_DTYPE_TO_NAME.get(dtype, str(dtype).replace("torch.", ""))


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _NAME_TO_TORCH_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype {dtype!r}") from exc


def canonical_torch_device(device: str | torch.device) -> str:
    return str(torch.device(device))


@dataclass(frozen=True)
class TorchTensorPayload:
    array: torch.Tensor
    layout: str
    dtype: str
    device: str


@dataclass(frozen=True)
class TorchRuntimeOutputs:
    tensors: tuple[TorchTensorPayload, ...]
    names: tuple[str, ...]


class TorchTensorRegistry:
    """Mutable named store for intermediate torch tensors during post-processing."""

    def __init__(self, tensors: dict[str, torch.Tensor] | None = None):
        self._tensors: dict[str, torch.Tensor] = dict(tensors) if tensors else {}

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self._tensors[name]
        except KeyError:
            available = sorted(self._tensors)
            raise KeyError(f"Tensor {name!r} not found in registry. Available: {available}")

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        self._tensors[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def keys(self) -> object:
        return self._tensors.keys()

    def __repr__(self) -> str:
        shapes = {k: f"{tuple(v.shape)}@{v.device}" for k, v in self._tensors.items()}
        return f"TorchTensorRegistry({shapes})"
