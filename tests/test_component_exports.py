from __future__ import annotations

import ml_pipes
import ml_pipes.core as core
import ml_pipes.inspection as inspection
import ml_pipes.onnx as onnx
import ml_pipes.standard as standard
import ml_pipes.tensor as tensor
import ml_pipes.vision as vision


def test_root_namespace_has_no_legacy_convenience_exports() -> None:
    assert not hasattr(ml_pipes, "Pipeline")
    assert not hasattr(ml_pipes, "Pick")
    assert not hasattr(ml_pipes, "Decode")
    assert not hasattr(ml_pipes, "Infer")
    assert not hasattr(ml_pipes, "InspectionResult")


def test_core_component_surface_is_curated() -> None:
    assert core.__all__ == [
        "Context",
        "Embed",
        "Inline",
        "Operator",
        "OperatorLike",
        "Pipeline",
        "PipelineDescription",
        "RegionCloser",
        "RegionOpener",
        "SHORT_CIRCUIT",
        "embed",
        "inline",
    ]


def test_standard_component_surface_is_curated() -> None:
    assert standard.__all__ == [
        "Batch",
        "CollectItems",
        "Distinct",
        "DistinctBy",
        "DropNull",
        "Filter",
        "FilterNotNull",
        "Gather",
        "LazyPerItem",
        "Map",
        "MapNotNull",
        "MapValue",
        "PerItem",
        "Pick",
        "Recall",
        "Scatter",
        "Select",
        "SideEffectOp",
        "Store",
        "StreamItems",
        "Take",
        "TakeWhile",
        "UnBatch",
        "WrapMappingInObject",
    ]


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


def test_vision_component_surface_is_curated() -> None:
    assert vision.__all__ == [
        "BlendImages",
        "ClampDensity",
        "ConvertBoxFormat",
        "ConvertColorSpace",
        "Decode",
        "DensityPrediction",
        "DensityToHeatmap",
        "Detections",
        "DrawBoxes",
        "DrawMasks",
        "FilterPredictions",
        "FilterPredictionsByArea",
        "FilterPredictionsByClass",
        "FilterPredictionsByScore",
        "FilterTensorsByClasses",
        "FilterTensorsByMasksArea",
        "FilterTensorsByScore",
        "ImagePayload",
        "LoadFile",
        "LogDetections",
        "MapPredictionsToObjects",
        "MasksToBoxes",
        "MeanMaskScores",
        "NMM",
        "NMS",
        "Normalize",
        "Prediction",
        "ProjectBoxes",
        "ProjectMasks",
        "ProjectRoIMasks",
        "ReconstructMasks",
        "Resize",
        "ResizeMasks",
        "ResizeTransform",
        "SaveImage",
        "Segmentations",
        "Stitch",
        "SumDensity",
        "Tile",
        "TileRect",
        "ToDensityPrediction",
        "ToDetections",
        "ToSegmentations",
        "WeightMasksByScores",
    ]


def test_onnx_component_surface_is_curated() -> None:
    assert onnx.__all__ == [
        "Distribute",
        "Extract",
        "Infer",
        "RuntimeOutputs",
    ]


def test_inspection_component_surface_is_curated() -> None:
    assert inspection.__all__ == [
        "GroupBlock",
        "HtmlRenderer",
        "ImageBlock",
        "InspectionResult",
        "InspectionSerializer",
        "OutputBlock",
        "PipelineInspector",
        "PlotRenderer",
        "Renderer",
        "StepView",
        "TextBlock",
    ]
