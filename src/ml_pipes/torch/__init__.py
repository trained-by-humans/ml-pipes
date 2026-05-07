from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise ImportError(
        "ml_pipes.torch requires the optional torch extra. Install it with `pip install ml-pipes[torch]`."
    ) from exc

from .ops import (
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
    TorchAsType,
    TorchCollate,
    TorchDistribute,
    TorchExtract,
    TorchInfer,
    TorchNMS,
)
from .types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry

__all__ = [
    "ToDevice",
    "ToNumpy",
    "ToNumpyRegistry",
    "ToTorch",
    "ToTorchRegistry",
    "TorchAsType",
    "TorchCollate",
    "TorchDistribute",
    "TorchExtract",
    "TorchInfer",
    "TorchNMS",
    "TorchRuntimeOutputs",
    "TorchTensorPayload",
    "TorchTensorRegistry",
]
