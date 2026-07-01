from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing_extensions import Self
else:  # pragma: no cover
    try:
        from typing import Self
    except ImportError:
        Self = TypeVar("Self")

__all__ = [
    "BoxPrediction",
    "ClassPrediction",
    "Detections",
    "FilterablePrediction",
    "ImagePayload",
    "Prediction",
    "PredictionIndices",
    "PredictionMask",
    "ResizeTransform",
    "ScorePrediction",
    "Segmentations",
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


PredictionMask: TypeAlias = Sequence[bool] | npt.NDArray[np.bool_]
PredictionIndices: TypeAlias = Sequence[int] | npt.NDArray[np.integer[Any]]


class FilterablePrediction(Protocol):
    def filter(self, mask: PredictionMask) -> Self:
        ...


class ClassPrediction(FilterablePrediction, Protocol):
    classes: Sequence[int]


class ScorePrediction(FilterablePrediction, Protocol):
    scores: Sequence[float]


class BoxPrediction(FilterablePrediction, Protocol):
    boxes: Sequence[Sequence[float]]


@dataclass
class Prediction:
    """Base class for all typed prediction outputs."""

    def filter(self, mask: PredictionMask) -> Self:
        kept: list[int] = []
        for i, keep in enumerate(mask):
            if not isinstance(keep, (bool, np.bool_)):
                raise TypeError("Prediction.filter expects a boolean mask; use select() for integer indices.")
            if bool(keep):
                kept.append(i)
        return self._slice(kept)

    def select(self, indices: PredictionIndices) -> Self:
        kept: list[int] = []
        for index in indices:
            if isinstance(index, (bool, np.bool_)):
                raise TypeError("Prediction.select expects integer indices; use filter() for boolean masks.")
            kept.append(int(index))
        return self._slice(kept)

    def _slice(self, kept: Sequence[int]) -> Self:
        sliced = {
            field.name: [getattr(self, field.name)[i] for i in kept]
            for field in dataclasses.fields(self)
        }
        return type(self)(**sliced)


@dataclass
class Detections(Prediction):
    boxes: Sequence[Sequence[float]]
    scores: Sequence[float]
    classes: Sequence[int]


@dataclass
class Segmentations(Detections):
    masks: Sequence[np.ndarray]
