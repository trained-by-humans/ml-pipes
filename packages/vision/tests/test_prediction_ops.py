from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ml_pipes.vision import (
    Detections,
    FilterPredictions,
    FilterPredictionsByArea,
    FilterPredictionsByClass,
    FilterPredictionsByScore,
    ImagePayload,
    MapPredictionsToObjects,
    SaveImage,
    Segmentations,
)


@dataclass
class _NormalizedDetections(Detections):
    def __post_init__(self) -> None:
        self.boxes = tuple(tuple(box) for box in self.boxes)
        self.scores = tuple(self.scores)
        self.classes = tuple(self.classes)


def _detections(**kwargs) -> Detections:
    defaults = dict(boxes=[[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], scores=[0.9, 0.5, 0.8], classes=[0, 1, 0])
    defaults.update(kwargs)
    return Detections(**defaults)


def _segmentations() -> Segmentations:
    masks = [np.zeros((4, 4), dtype=bool) for _ in range(3)]
    return Segmentations(boxes=[[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], scores=[0.9, 0.5, 0.8], classes=[0, 1, 0], masks=masks)


def test_save_image_writes_output(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "annotated.jpg"

    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
    result = SaveImage(output_path)(payload)

    assert output_path.is_file()
    assert result is payload


def test_save_image_at_zero_writes_output_and_returns_tuple(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "annotated.jpg"

    payload = (
        ImagePayload(array=image, color_space="BGR", layout="HWC"),
        {"meta": 1},
    )
    result = SaveImage(output_path, at=0)(payload)

    assert output_path.is_file()
    assert result is payload


def test_map_to_objects_can_convert_detection_result():
    detections = Detections(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )

    result = MapPredictionsToObjects(
        fields={
            "box": "boxes",
            "score": "scores",
            "class_id": "classes",
        }
    )(detections)

    assert result == [
        {
            "box": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
            "class_id": 1,
        }
    ]


def test_map_to_objects_at_one_replaces_prediction_slot():
    payload = (
        "prefix",
        Detections(
            boxes=[[1.0, 2.0, 3.0, 4.0]],
            scores=[0.9],
            classes=[1],
        ),
    )

    result = MapPredictionsToObjects(
        fields={
            "box": "boxes",
            "score": "scores",
            "class_id": "classes",
        },
        at=1,
    )(payload)

    assert result == (
        "prefix",
        [
            {
                "box": [1.0, 2.0, 3.0, 4.0],
                "score": 0.9,
                "class_id": 1,
            }
        ],
    )


def test_map_to_objects_supports_segmentation_callbacks():
    segmentations = Segmentations(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
        masks=[np.array([[True, False], [True, True]], dtype=bool)],
    )
    mapper = MapPredictionsToObjects(
        fields={
            "class_id": "classes",
            "area": lambda prediction: [int(np.asarray(mask, dtype=bool).sum()) for mask in prediction.masks],
        },
    )

    result = mapper(segmentations)

    assert result == [{"class_id": 1, "area": 3}]


def test_map_to_objects_requires_equal_length_columns():
    detections = Detections(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )

    with pytest.raises(ValueError, match="equal-length collections"):
        MapPredictionsToObjects(
            fields={
                "box": "boxes",
                "score": lambda prediction: prediction.scores + [0.1],
            }
        )(detections)


def test_map_to_objects_with_at_requires_tuple_payload():
    detections = Detections(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )

    with pytest.raises(TypeError, match="requires a tuple payload"):
        MapPredictionsToObjects(fields={"box": "boxes"}, at=1)(detections)


def test_filter_predictions_generic_predicate():
    detections = _detections()
    result = FilterPredictions(predicate=lambda prediction: [class_id == 0 for class_id in prediction.classes])(detections)

    assert result.classes == [0, 0]
    assert result.scores == [0.9, 0.8]


def test_filter_predictions_preserves_subclass_type():
    segmentations = _segmentations()
    result = FilterPredictions(predicate=lambda prediction: [class_id == 0 for class_id in prediction.classes])(segmentations)

    assert type(result) is Segmentations


def test_filter_predictions_slices_all_fields():
    segmentations = _segmentations()
    result = FilterPredictions(predicate=lambda prediction: [class_id == 0 for class_id in prediction.classes])(segmentations)

    assert len(result.masks) == 2
    assert len(result.boxes) == 2


def test_filter_predictions_by_class():
    detections = _detections()
    result = FilterPredictionsByClass({0})(detections)

    assert result.classes == [0, 0]


def test_filter_predictions_by_score():
    detections = _detections()
    result = FilterPredictionsByScore(min_score=0.7)(detections)

    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_filter_predictions_by_area_min():
    detections = Detections(boxes=[[0, 0, 5, 5], [0, 0, 1, 1]], scores=[0.9, 0.8], classes=[0, 0])
    result = FilterPredictionsByArea(min_area=10)(detections)

    assert len(result.boxes) == 1
    assert result.boxes[0] == [0, 0, 5, 5]


def test_filter_predictions_by_area_max():
    detections = Detections(boxes=[[0, 0, 5, 5], [0, 0, 1, 1]], scores=[0.9, 0.8], classes=[0, 0])
    result = FilterPredictionsByArea(max_area=2)(detections)

    assert len(result.boxes) == 1
    assert result.boxes[0] == [0, 0, 1, 1]


def test_filter_bool_list():
    detections = _detections()
    result = detections.filter([True, False, True])

    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_filter_bool_list_all_false():
    detections = _detections()
    result = detections.filter([False, False, False])

    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_filter_bool_list_all_true():
    detections = _detections()
    result = detections.filter([True, True, True])

    assert result.scores == detections.scores
    assert result.classes == detections.classes


def test_filter_numpy_bool_array():
    detections = _detections()
    mask = np.array([True, False, True])
    result = detections.filter(mask)

    assert result.scores == [0.9, 0.8]


def test_select_index_list():
    detections = _detections()
    result = detections.select([0, 2])

    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_select_index_list_single():
    detections = _detections()
    result = detections.select([1])

    assert result.scores == [0.5]
    assert result.classes == [1]


def test_select_numpy_index_array():
    detections = _detections()
    result = detections.select(np.array([0, 2]))

    assert result.scores == [0.9, 0.8]


def test_select_numpy_argsort():
    detections = _detections()
    indices = np.argsort(detections.scores)[-2:]
    result = detections.select(indices)

    assert set(result.scores) == {0.9, 0.8}


def test_filter_reconstructs_prediction_subclass_via_init():
    detections = _NormalizedDetections(
        boxes=[[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
        scores=[0.9, 0.5, 0.8],
        classes=[0, 1, 0],
    )
    result = detections.filter([True, False, True])

    assert isinstance(result, _NormalizedDetections)
    assert result.boxes == ((0, 0, 1, 1), (2, 2, 3, 3))
    assert result.scores == (0.9, 0.8)
    assert result.classes == (0, 0)


def test_select_reconstructs_prediction_subclass_via_init():
    detections = _NormalizedDetections(
        boxes=[[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
        scores=[0.9, 0.5, 0.8],
        classes=[0, 1, 0],
    )
    result = detections.select([0, 2])

    assert isinstance(result, _NormalizedDetections)
    assert result.boxes == ((0, 0, 1, 1), (2, 2, 3, 3))
    assert result.scores == (0.9, 0.8)
    assert result.classes == (0, 0)


def test_filter_empty_mask():
    detections = _detections()
    result = detections.filter([])

    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_filter_integer_indices_raise():
    detections = _detections()

    with pytest.raises(TypeError):
        detections.filter([0, 2])


def test_select_boolean_mask_raise():
    detections = _detections()

    with pytest.raises(TypeError):
        detections.select([True, False])


def test_select_index_out_of_bounds_raises():
    detections = _detections()

    with pytest.raises(IndexError):
        detections.select([0, 99])


def test_filter_bool_mask_too_long_raises():
    detections = _detections()

    with pytest.raises(IndexError):
        detections.filter([True, False, True, True])
