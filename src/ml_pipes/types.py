from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt


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
class TensorPayload:
    array: np.ndarray
    layout: str
    dtype: str


@dataclass(frozen=True)
class RuntimeOutputs:
    # Runtime-facing output tensors exactly as exposed by the exported graph.
    tensors: tuple[TensorPayload, ...]
    names: tuple[str, ...]


class TensorRegistry:
    """Mutable named store for intermediate tensors during post-processing."""

    def __init__(self, tensors: dict[str, np.ndarray] | None = None):
        self._tensors: dict[str, np.ndarray] = dict(tensors) if tensors else {}

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._tensors[name]
        except KeyError:
            available = sorted(self._tensors)
            raise KeyError(f"Tensor {name!r} not found in registry. Available: {available}")

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        self._tensors[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def keys(self) -> object:
        return self._tensors.keys()

    def __repr__(self) -> str:
        shapes = {k: v.shape for k, v in self._tensors.items()}
        return f"TensorRegistry({shapes})"


@dataclass(frozen=True)
class ResizeTransform:
    scale: tuple[float, float]
    pad: tuple[float, float]
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]


PredictionT = TypeVar("PredictionT", bound="Prediction")
PredictionMask: TypeAlias = Sequence[bool] | npt.NDArray[np.bool_]
PredictionIndices: TypeAlias = Sequence[int] | npt.NDArray[np.integer[Any]]


@dataclass(frozen=True)
class Prediction:
    """Base class for all typed prediction outputs.

    All fields must be equal-length lists (one entry per detected instance).
    The filter and select methods slice every field uniformly, so they work
    generically for any subclass without knowing its field names.
    """

    def filter(self: PredictionT, mask: PredictionMask) -> PredictionT:
        kept = [i for i, keep in enumerate(mask) if bool(keep)]
        return self._slice(kept)

    def select(self: PredictionT, indices: PredictionIndices) -> PredictionT:
        kept = [int(index) for index in indices]
        return self._slice(kept)

    def _slice(self: PredictionT, kept: Sequence[int]) -> PredictionT:
        sliced = {
            f.name: [getattr(self, f.name)[i] for i in kept]
            for f in dataclasses.fields(self)
        }
        return type(self)(**sliced)


@dataclass(frozen=True)
class Detections(Prediction):
    boxes: list[list[float]]
    scores: list[float]
    classes: list[int]


@dataclass(frozen=True)
class Segmentations(Detections):
    masks: list[np.ndarray]
