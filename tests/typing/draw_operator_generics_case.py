from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from ml_pipes.ops import DrawBoxes, DrawMasks
from ml_pipes.types import Detections, ImagePayload, Segmentations


@dataclass
class RichDetections(Detections):
    labels: list[str]


source = ImagePayload(array=np.zeros((8, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
rich_detections = RichDetections(
    boxes=[[1.0, 1.0, 6.0, 6.0]],
    scores=[0.9],
    classes=[1],
    labels=["person"],
)
segmentations = Segmentations(
    boxes=[[1.0, 1.0, 6.0, 6.0]],
    scores=[0.9],
    classes=[1],
    masks=[np.zeros((8, 8), dtype=bool)],
)

assert_type(DrawBoxes()(source, rich_detections), tuple[ImagePayload, RichDetections])
assert_type(DrawBoxes()(source, segmentations), tuple[ImagePayload, Segmentations])
assert_type(DrawMasks()(source, segmentations), tuple[ImagePayload, Segmentations])
