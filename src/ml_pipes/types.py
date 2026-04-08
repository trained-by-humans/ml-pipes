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


@dataclass(frozen=True)
class DetectionBatch:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    boxes: list[list[float]]
    scores: list[float]
    classes: list[int]
