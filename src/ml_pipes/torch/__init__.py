from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise ImportError(
        "ml_pipes.torch requires the optional torch extra. Install it with `pip install ml-pipes[torch]`."
    ) from exc

from .ops import (
    TorchArgMax,
    TorchGatherScores,
    TorchSigmoid,
    TorchSlice,
    TorchSoftmax,
    TorchThresholdTensors,
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
    TorchMultiplyTensors,
    TorchNMS,
    TorchWeightMasksByScores,
)
from .types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry

__all__ = [
    "ToDevice",
    "TorchArgMax",
    "TorchGatherScores",
    "TorchSigmoid",
    "TorchSlice",
    "TorchSoftmax",
    "TorchThresholdTensors",
    "ToNumpy",
    "ToNumpyRegistry",
    "ToTorch",
    "ToTorchRegistry",
    "TorchAsType",
    "TorchCollate",
    "TorchDistribute",
    "TorchExtract",
    "TorchInfer",
    "TorchMultiplyTensors",
    "TorchNMS",
    "TorchWeightMasksByScores",
    "TorchRuntimeOutputs",
    "TorchTensorPayload",
    "TorchTensorRegistry",
]
