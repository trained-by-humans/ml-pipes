from __future__ import annotations

import cv2
import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.tensor import TensorRegistry
from .types import ImagePayload, ResizeTransform


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
class ProjectDensity:
    def __init__(
        self,
        src: str = "density",
        as_: str | None = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        self.src = src
        self.as_ = as_ or src
        self.interpolation = interpolation

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        density = registry[self.src]
        if density.ndim != 2:
            raise ValueError(f"ProjectDensity expects a 2D density tensor, got shape {density.shape}")

        model_height, model_width = transform.resized_shape
        source_height, source_width = transform.original_shape
        scale_x, scale_y = transform.scale
        pad_x, pad_y = transform.pad
        content_height = int(round(source_height * scale_y))
        content_width = int(round(source_width * scale_x))

        density_height, density_width = density.shape
        top = int(round(pad_y * density_height / model_height))
        left = int(round(pad_x * density_width / model_width))
        bottom = int(round((pad_y + content_height) * density_height / model_height))
        right = int(round((pad_x + content_width) * density_width / model_width))
        if top < 0 or left < 0 or bottom > density_height or right > density_width or top >= bottom or left >= right:
            raise ValueError(
                "ProjectDensity received a ResizeTransform whose content region does not fit the density tensor"
            )

        content_density = density[top:bottom, left:right]
        projected = cv2.resize(
            content_density.astype(np.float32, copy=False),
            (source_width, source_height),
            interpolation=self.interpolation,
        )
        content_sum = float(content_density.sum())
        projected_sum = float(projected.sum())
        if projected_sum != 0.0:
            projected *= content_sum / projected_sum
        registry[self.as_] = projected
        return registry


@Operator
class DensityToHeatmap:
    def __init__(
        self,
        src: str = "density",
        colormap: int | None = cv2.COLORMAP_TURBO,
    ) -> None:
        self.src = src
        self.colormap = colormap

    def __call__(self, registry: TensorRegistry) -> ImagePayload:
        density = np.maximum(registry[self.src], 0)
        if density.ndim != 2:
            raise ValueError(f"DensityToHeatmap expects a 2D density tensor, got shape {density.shape}")
        height, width = density.shape
        if density.size == 0 or float(density.max()) <= 0.0:
            normalized = np.zeros((height, width), dtype=np.uint8)
        else:
            normalized = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
        if self.colormap is None:
            return ImagePayload(array=normalized, color_space="GRAY", layout="HW")
        heatmap = cv2.applyColorMap(normalized, self.colormap)
        return ImagePayload(array=heatmap, color_space="BGR", layout="HWC")
