from __future__ import annotations

from .detection import (
    ConvertBoxFormat,
    DrawBoxes,
    FilterTensorsByBoxArea,
    FilterTensorsByClasses,
    FilterTensorsByScore,
    LogDetections,
    NMM,
    NMS,
    ProjectBoxes,
)
from .density import ClampDensity, DensityToHeatmap, ProjectDensity, SumDensity
from .ops import (
    BlendImages,
    ConvertColorSpace,
    Decode,
    LoadFile,
    Normalize,
    Resize,
    SaveImage,
)
from .segmentation import (
    DrawMasks,
    FilterTensorsByMasksArea,
    MasksToBoxes,
    MeanMaskScores,
    ProjectMasks,
    ProjectRoIMasks,
    ReconstructMasks,
    ResizeMasks,
    WeightMasksByScores,
)
from .tiling import Stitch, Tile, TileRect
from .types import ImagePayload, ResizeTransform

__all__ = [
    "BlendImages",
    "ClampDensity",
    "ConvertBoxFormat",
    "ConvertColorSpace",
    "Decode",
    "DensityToHeatmap",
    "DrawBoxes",
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

from ._inspection import register_inspection_formatters as _register_inspection_formatters


_register_inspection_formatters()
