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
    TorchSynchronizeTensors,
)
from .runtime_ops import (
    TorchCollate,
    TorchDistribute,
    TorchExtract,
    TorchInfer,
)
from .tensor_ops import (
    TorchArgMax,
    TorchApplyTensorMask,
    TorchAsType,
    TorchBinarizeTensor,
    TorchBinarizeTensorByThreshold,
    TorchCreateTensorMask,
    TorchCreateTensorMaskByThreshold,
    TorchFilterTensors,
    TorchGatherRows,
    TorchGatherScores,
    TorchMapTensor,
    TorchMultiplyTensors,
    TorchScale,
    TorchSelectTensors,
    TorchSigmoid,
    TorchSqueeze,
    TorchSlice,
    TorchSortTensorsBy,
    TorchSoftmax,
    TorchTopK,
    TorchTopKIndices2D,
    TorchTranspose,
)
from .types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry
from .vision_ops import (
    TorchConvertBoxFormat,
    TorchFilterTensorsByScore,
    TorchFilterTensorsByClasses,
    TorchFilterTensorsByMasksArea,
    TorchWeightMasksByScores,
    TorchResizeMasks,
    TorchMeanMaskedScores,
    TorchMasksToBoxes,
    TorchReconstructMasks,
    TorchNMS,
)

__all__ = [
    "ToDevice",
    "ToNumpy",
    "ToNumpyRegistry",
    "ToTorch",
    "ToTorchRegistry",
    "TorchSynchronizeTensors",
    "TorchCollate",
    "TorchDistribute",
    "TorchExtract",
    "TorchInfer",
    "TorchAsType",
    "TorchArgMax",
    "TorchApplyTensorMask",
    "TorchBinarizeTensor",
    "TorchBinarizeTensorByThreshold",
    "TorchCreateTensorMask",
    "TorchCreateTensorMaskByThreshold",
    "TorchFilterTensors",
    "TorchGatherRows",
    "TorchGatherScores",
    "TorchMapTensor",
    "TorchMultiplyTensors",
    "TorchScale",
    "TorchSelectTensors",
    "TorchSigmoid",
    "TorchSqueeze",
    "TorchSlice",
    "TorchSortTensorsBy",
    "TorchSoftmax",
    "TorchTopK",
    "TorchTopKIndices2D",
    "TorchTranspose",
    "TorchConvertBoxFormat",
    "TorchFilterTensorsByScore",
    "TorchFilterTensorsByClasses",
    "TorchFilterTensorsByMasksArea",
    "TorchWeightMasksByScores",
    "TorchResizeMasks",
    "TorchMeanMaskedScores",
    "TorchMasksToBoxes",
    "TorchReconstructMasks",
    "TorchNMS",
    "TorchRuntimeOutputs",
    "TorchTensorPayload",
    "TorchTensorRegistry",
]

from ._inspection import register_inspection_formatters as _register_inspection_formatters


_register_inspection_formatters()
