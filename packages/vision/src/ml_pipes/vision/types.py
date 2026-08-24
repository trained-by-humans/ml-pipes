from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = [
    "ImagePayload",
    "ResizeTransform",
]


@dataclass(frozen=True)
class ImagePayload:
    array: np.ndarray
    color_space: str = "BGR"
    layout: str = "HWC"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.array.shape)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        try:
            h_axis = self.layout.index("H")
            w_axis = self.layout.index("W")
        except ValueError as exc:
            raise ValueError(f"ImagePayload layout must contain H and W, got {self.layout!r}") from exc
        return int(self.array.shape[h_axis]), int(self.array.shape[w_axis])

    @property
    def height(self) -> int:
        return self.spatial_shape[0]

    @property
    def width(self) -> int:
        return self.spatial_shape[1]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)

    @property
    def ndim(self) -> int:
        return int(self.array.ndim)

    @property
    def channels(self) -> int | None:
        if "C" not in self.layout:
            return None
        return int(self.array.shape[self.layout.index("C")])


@dataclass(frozen=True)
class ResizeTransform:
    scale: tuple[float, float]
    pad: tuple[float, float]
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
