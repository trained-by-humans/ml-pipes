from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

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


def test_core_import_does_not_require_inspection_extras() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {"cv2", "ml_pipes.onnx", "ml_pipes.tensor", "ml_pipes.vision"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or any(fullname.startswith(name + ".") for name in blocked):
                    raise ModuleNotFoundError(f"No module named {fullname!r}")
                return None

        sys.meta_path.insert(0, Blocker())

        import ml_pipes.core

        unexpected = sorted(
            name
            for name in sys.modules
            if name in {
                "ml_pipes.inspection.formatters",
                "ml_pipes.inspection.html_renderer",
                "ml_pipes.inspection.plot_renderer",
                "ml_pipes.inspection.inspector",
                "ml_pipes.inspection.views",
            }
            or name in blocked
            or any(name.startswith(blocked_name + ".") for blocked_name in blocked)
        )
        assert not unexpected, unexpected
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


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
