from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .operator import Operator
from .types import ImagePayload, TensorRegistry


@dataclass(frozen=True)
class DensityPrediction:
    density_map: np.ndarray


class ClampDensity:
    def __call__(self, prediction: DensityPrediction) -> DensityPrediction:
        return DensityPrediction(
            density_map=np.maximum(prediction.density_map, 0),
        )


class SumDensity:
    def __call__(self, prediction: DensityPrediction) -> float:
        return float(prediction.density_map.sum())


@Operator
class ToDensityPrediction:
    def __init__(self, src: str = "density") -> None:
        self.src = src

    def __call__(self, registry: TensorRegistry) -> DensityPrediction:
        return DensityPrediction(
            density_map=np.asarray(registry[self.src]),
        )


@Operator
class DensityToHeatmap:
    def __init__(
        self,
        colormap: int = cv2.COLORMAP_TURBO,
        interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        self.colormap = colormap
        self.interpolation = interpolation

    def __call__(self, source_image: ImagePayload, prediction: DensityPrediction) -> tuple[ImagePayload, ImagePayload]:
        height, width = source_image.array.shape[:2]
        density = np.maximum(prediction.density_map, 0)
        if density.size == 0 or float(density.max()) <= 0.0:
            heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            normalized = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(normalized.astype(np.uint8), self.colormap)
            heatmap = cv2.resize(heatmap, (width, height), interpolation=self.interpolation)
        return source_image, ImagePayload(array=heatmap, color_space="BGR", layout="HWC")


@Operator
class BlendImages:
    def __init__(self, base_weight: float = 0.60, overlay_weight: float = 0.40) -> None:
        self.base_weight = base_weight
        self.overlay_weight = overlay_weight

    def __call__(self, source_image: ImagePayload, overlay_image: ImagePayload) -> ImagePayload:
        if source_image.layout != "HWC" or overlay_image.layout != "HWC":
            raise ValueError("BlendImages expects HWC images")
        if source_image.array.shape != overlay_image.array.shape:
            raise ValueError(
                f"BlendImages requires matching shapes, got {source_image.array.shape} and {overlay_image.array.shape}"
            )
        blended = cv2.addWeighted(
            source_image.array,
            self.base_weight,
            overlay_image.array,
            self.overlay_weight,
            0.0,
        )
        return ImagePayload(
            array=blended,
            color_space=source_image.color_space,
            layout=source_image.layout,
        )
