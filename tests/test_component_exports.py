from __future__ import annotations

import ml_pipes
import ml_pipes.core as core
import ml_pipes.inspection as inspection
import ml_pipes.onnx as onnx
import ml_pipes.standard as standard
import ml_pipes.tensor as tensor
import ml_pipes.vision as vision


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
    assert core.Pipeline is ml_pipes.Pipeline
    assert core.Operator is ml_pipes.Operator


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
    assert standard.Pick is ml_pipes.Pick
    assert standard.Store is ml_pipes.Store


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
    assert tensor.TensorPayload is ml_pipes.TensorPayload
    assert tensor.ArgMax is ml_pipes.ArgMax


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
    assert vision.Decode is ml_pipes.Decode
    assert vision.ImagePayload is ml_pipes.ImagePayload


def test_onnx_component_surface_is_curated() -> None:
    assert onnx.__all__ == [
        "Distribute",
        "Extract",
        "Infer",
        "RuntimeOutputs",
    ]
    assert onnx.Infer is ml_pipes.Infer
    assert onnx.RuntimeOutputs is ml_pipes.RuntimeOutputs


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
    assert inspection.InspectionResult is ml_pipes.InspectionResult
    assert inspection.PipelineInspector is ml_pipes.PipelineInspector
