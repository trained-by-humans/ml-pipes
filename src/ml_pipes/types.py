from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImagePayload:
    array: np.ndarray
    color_space: str = "BGR"
    layout: str = "HWC"


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


@dataclass(frozen=True)
class Detections:
    boxes: list[list[float]]
    scores: list[float]
    classes: list[int]


@dataclass(frozen=True)
class Segmentations:
    boxes: list[list[float]]
    scores: list[float]
    classes: list[int]
    masks: list[np.ndarray]
