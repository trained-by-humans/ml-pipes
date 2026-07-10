from __future__ import annotations

import ml_pipes.tensor as tensor


def test_tensor_component_surface_is_curated() -> None:
    assert tensor.__all__ == [
        "ApplyTensorMask",
        "ArgMax",
        "AsType",
        "BinarizeTensor",
        "BinarizeTensorByThreshold",
        "Collate",
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
        "Slice",
        "Softmax",
        "SortTensorsBy",
        "Squeeze",
        "TensorPayload",
        "TensorRegistry",
        "TopK",
        "TopKIndices2D",
        "Transpose",
    ]


def test_tensor_alias_exports_preserve_identity() -> None:
    assert tensor.GatherScores is tensor.GatherRows
    assert tensor.BinarizeTensor is tensor.CreateTensorMask
    assert tensor.BinarizeTensorByThreshold is tensor.CreateTensorMaskByThreshold
