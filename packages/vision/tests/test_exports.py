from __future__ import annotations

import ml_pipes.vision as vision


def test_vision_component_surface_is_curated() -> None:
    assert vision.__all__ == [
        "BlendImages",
        "ClampDensity",
        "ConvertBoxFormat",
        "ConvertColorSpace",
        "Decode",
        "DensityToHeatmap",
        "DrawBoxes",
        "DrawDensityOverlay",
        "DrawMasks",
        "FilterTensorsByBoxArea",
        "FilterTensorsByClasses",
        "FilterTensorsByMasksArea",
        "FilterTensorsByScore",
        "ImagePayload",
        "LoadFile",
        "LogDetections",
        "MasksToBoxes",
        "MeanMaskScores",
        "NMM",
        "NMS",
        "Normalize",
        "ProjectBoxes",
        "ProjectDensity",
        "ProjectMasks",
        "ProjectRoIMasks",
        "ReconstructMasks",
        "Resize",
        "ResizeMasks",
        "ResizeTransform",
        "SaveImage",
        "Stitch",
        "SumDensity",
        "Tile",
        "TileRect",
        "WeightMasksByScores",
    ]
