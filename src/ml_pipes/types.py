from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np


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


@dataclass(frozen=True)
class Prediction:
    """Base class for all typed prediction outputs.

    All fields must be equal-length lists (one entry per detected instance).
    The filter method slices every field by index, so it works generically for
    any subclass without knowing its field names.
    """

    def filter(self: PredictionT, mask: Any) -> PredictionT:
        if len(mask) == 0:
            kept: list[int] = []
        elif isinstance(mask[0], (bool, np.bool_)):
            kept = [i for i, m in enumerate(mask) if m]
        else:
            kept = [int(i) for i in mask]
        sliced = {f.name: [getattr(self, f.name)[i] for i in kept]
                  for f in dataclasses.fields(self)}
        return type(self)(**sliced)


@dataclass(frozen=True)
class Detections(Prediction):
    boxes: list[list[float]]
    scores: list[float]
    classes: list[int]


@dataclass(frozen=True)
class Segmentations(Detections):
    masks: list[np.ndarray]
