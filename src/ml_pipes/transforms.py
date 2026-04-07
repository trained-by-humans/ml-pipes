from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResizeTransform:
    scale: float
    pad: tuple[float, float]
    original_shape: tuple[int, int]
