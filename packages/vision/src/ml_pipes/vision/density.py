from __future__ import annotations

import cv2
import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.tensor import TensorRegistry
from .types import ImagePayload


@Operator
class ClampDensity:
    def __init__(self, src: str = "density", as_: str | None = None) -> None:
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = np.maximum(registry[self.src], 0)
        return registry


@Operator
class SumDensity:
    def __init__(self, src: str = "density") -> None:
        self.src = src

    def __call__(self, registry: TensorRegistry) -> float:
        return float(registry[self.src].sum())


@Operator
class DensityToHeatmap:
    def __init__(
        self,
        src: str = "density",
        colormap: int = cv2.COLORMAP_TURBO,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self.src = src
        self.colormap = colormap
        self.interpolation = interpolation

    def __call__(self, source_image: ImagePayload, registry: TensorRegistry) -> tuple[ImagePayload, ImagePayload]:
        height, width = source_image.array.shape[:2]
        density = np.maximum(registry[self.src], 0)
        if density.size == 0 or float(density.max()) <= 0.0:
            heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            normalized = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(normalized.astype(np.uint8), self.colormap)
            heatmap = cv2.resize(heatmap, (width, height), interpolation=self.interpolation)
        return source_image, ImagePayload(array=heatmap, color_space="BGR", layout="HWC")
