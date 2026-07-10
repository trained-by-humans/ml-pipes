from __future__ import annotations

import ml_pipes.core as core
import ml_pipes.inspection as inspection
import ml_pipes.standard as standard


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
