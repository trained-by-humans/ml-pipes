from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise ImportError(
        "ml_pipes.torch requires the optional torch extra. Install it with `pip install ml-pipes[torch]`."
    ) from exc

from .boundary_ops import (
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
    SynchronizeTensors,
)
from .runtime_ops import (
    Collate,
    Distribute,
    Extract,
    Infer,
)
from .tensor_ops import (
    ArgMax,
    ApplyTensorMask,
    AsType,
    BinarizeTensor,
    BinarizeTensorByThreshold,
    CreateTensorMask,
    CreateTensorMaskByThreshold,
    FilterTensors,
    GatherRows,
    GatherScores,
    MapTensor,
    MultiplyTensors,
    Scale,
    SelectTensors,
    Sigmoid,
    Squeeze,
    Slice,
    SortTensorsBy,
    Softmax,
    TopK,
    TopKIndices2D,
    Transpose,
)
from .types import RuntimeOutputs, TensorPayload, TensorRegistry

__all__ = [
    "ToDevice",
    "ToNumpy",
    "ToNumpyRegistry",
    "ToTorch",
    "ToTorchRegistry",
    "SynchronizeTensors",
    "Collate",
    "Distribute",
    "Extract",
    "Infer",
    "AsType",
    "ArgMax",
    "ApplyTensorMask",
    "BinarizeTensor",
    "BinarizeTensorByThreshold",
    "CreateTensorMask",
    "CreateTensorMaskByThreshold",
    "FilterTensors",
    "GatherRows",
    "GatherScores",
    "MapTensor",
    "MultiplyTensors",
    "Scale",
    "SelectTensors",
    "Sigmoid",
    "Squeeze",
    "Slice",
    "SortTensorsBy",
    "Softmax",
    "TopK",
    "TopKIndices2D",
    "Transpose",
    "RuntimeOutputs",
    "TensorPayload",
    "TensorRegistry",
]

from ._inspection import register_inspection_formatters as _register_inspection_formatters


_register_inspection_formatters()
