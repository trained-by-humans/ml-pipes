from __future__ import annotations

import numpy as np

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from typing import Literal

from ml_pipes.vision import MapPredictionsToObjects
from ml_pipes.vision import (
    Detections,
    ImagePayload,
    Segmentations,
)


sample_image = ImagePayload(array=np.zeros((8, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
sample_detections = Detections(
    boxes=[[1.0, 1.0, 6.0, 6.0]],
    scores=[0.9],
    classes=[1],
)
sample_segmentations = Segmentations(
    boxes=[[1.0, 1.0, 6.0, 6.0]],
    scores=[0.9],
    classes=[1],
    masks=[np.zeros((8, 8), dtype=bool)],
)

def segmentation_indices(prediction: Segmentations) -> list[int]:
    return list(range(len(prediction.classes)))


def segmentation_areas(prediction: Segmentations) -> list[int]:
    return [int(np.asarray(mask, dtype=bool).sum()) for mask in prediction.masks]


segmentation_mapper = MapPredictionsToObjects[Literal[1], Segmentations](
    fields={
        "index": segmentation_indices,
        "area": segmentation_areas,
        "box": "boxes",
    },
    at=1,
)

assert_type(
    segmentation_mapper((sample_image, sample_segmentations)),
    tuple[ImagePayload, list[dict[str, object]]],
)
assert_type(
    MapPredictionsToObjects[None, Segmentations](
        fields={
            "class_id": "classes",
            "area": segmentation_areas,
        },
    )(sample_segmentations),
    list[dict[str, object]],
)
assert_type(
    MapPredictionsToObjects[None, Detections](fields={"box": "boxes"})(sample_detections),
    list[dict[str, object]],
)
