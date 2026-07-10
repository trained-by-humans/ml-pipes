from __future__ import annotations

import io

import numpy as np

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import (
    ConvertBoxFormat,
    Detections,
    DrawBoxes,
    FilterTensorsByClasses,
    FilterTensorsByScore,
    ImagePayload,
    LogDetections,
    NMS,
    ProjectBoxes,
    ResizeTransform,
    Segmentations,
    ToDetections,
)


def _make_registry(boxes, scores, classes):
    registry = TensorRegistry()
    registry["boxes"] = np.array(boxes, dtype=np.float32)
    registry["scores"] = np.array(scores, dtype=np.float32)
    registry["classes"] = np.array(classes, dtype=np.int32)
    return registry


def test_filter_tensors_by_score_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "scores": np.array([0.9, 0.5, 0.8], dtype=np.float32),
            "classes": np.array([0, 1, 0], dtype=np.int64),
        }
    )

    result = FilterTensorsByScore(
        "classes",
        score="scores",
        min_score=0.75,
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert np.allclose(result["selected_scores"], [0.9, 0.8])
    assert result["selected_classes"].tolist() == [0, 0]
    assert np.allclose(result["scores"], [0.9, 0.5, 0.8])


def test_filter_tensors_by_classes_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "scores": np.array([0.9, 0.5, 0.8], dtype=np.float32),
            "classes": np.array([0, 1, 2], dtype=np.int64),
        }
    )

    result = FilterTensorsByClasses(
        "scores",
        classes="classes",
        keep_classes=[0, 2],
        as_=("selected_classes", "selected_scores"),
    )(registry)

    assert result["selected_classes"].tolist() == [0, 2]
    assert np.allclose(result["selected_scores"], [0.9, 0.8])
    assert result["classes"].tolist() == [0, 1, 2]


def test_convert_box_format_cxcywh_to_xyxy():
    registry = TensorRegistry({"boxes": np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)})

    result = ConvertBoxFormat(from_="cxcywh")(registry)

    assert np.allclose(result["boxes"], [[8.0, 17.0, 12.0, 23.0]])


def test_nms_keeps_overlapping_boxes_from_different_classes():
    registry = _make_registry(
        boxes=[[10, 10, 50, 50], [12, 12, 48, 48]],
        scores=[0.95, 0.9],
        classes=[0, 1],
    )

    result = NMS()(registry)

    assert result["boxes"].shape == (2, 4)
    assert result["classes"].tolist() == [0, 1]


def test_nms_suppresses_same_class_overlap():
    registry = _make_registry(
        boxes=[[10, 10, 50, 50], [12, 12, 48, 48], [100, 100, 140, 140]],
        scores=[0.95, 0.85, 0.8],
        classes=[0, 0, 0],
    )

    result = NMS()(registry)

    assert result["boxes"].shape == (2, 4)
    assert np.allclose(result["scores"], [0.95, 0.8])


def test_nms_kept_as_records_original_kept_indices():
    registry = _make_registry(
        boxes=[[10, 10, 50, 50], [12, 12, 48, 48], [100, 100, 140, 140]],
        scores=[0.95, 0.85, 0.8],
        classes=[0, 0, 0],
    )

    result = NMS(kept_as="kept")(registry)

    assert result["kept"].tolist() == [0, 2]


def test_project_boxes_reverses_padding_and_scale():
    registry = _make_registry(
        boxes=[[30.0, 40.0, 110.0, 120.0]],
        scores=[0.9],
        classes=[3],
    )
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(10.0, 20.0),
        original_shape=(100, 200),
        resized_shape=(240, 420),
    )

    result = ProjectBoxes()(registry, transform)

    assert result["boxes"].tolist() == [[10.0, 10.0, 50.0, 50.0]]


def test_project_boxes_clips_to_original_bounds():
    registry = _make_registry(
        boxes=[[-50.0, -50.0, 500.0, 400.0]],
        scores=[0.9],
        classes=[1],
    )
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(10.0, 20.0),
        original_shape=(100, 200),
        resized_shape=(240, 420),
    )

    result = ProjectBoxes()(registry, transform)

    assert result["boxes"].tolist() == [[0.0, 0.0, 200.0, 100.0]]


def test_to_detections_converts_registry_to_detections():
    registry = _make_registry(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )

    result = ToDetections()(registry)

    assert isinstance(result, Detections)
    assert result.boxes == [[1.0, 2.0, 3.0, 4.0]]
    assert np.allclose(result.scores, [0.9])
    assert result.classes == [1]


def test_draw_boxes_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ImagePayload(array=image, color_space="BGR", layout="HWC")
    detections = Detections(
        boxes=[[4.0, 4.0, 20.0, 20.0]],
        scores=[0.9],
        classes=[1],
    )

    result, returned_detections = DrawBoxes(class_names=["zero", "one"], color=(0, 255, 0))(source, detections)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)
    assert returned_detections is detections


def test_draw_boxes_preserves_segmentations_instance():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ImagePayload(array=image, color_space="BGR", layout="HWC")
    segmentations = Segmentations(
        boxes=[[4.0, 4.0, 20.0, 20.0]],
        scores=[0.9],
        classes=[1],
        masks=[np.zeros((32, 32), dtype=bool)],
    )

    result, returned_segmentations = DrawBoxes(class_names=["zero", "one"], color=(0, 255, 0))(source, segmentations)

    assert result.array.shape == image.shape
    assert np.any(result.array != 0)
    assert returned_segmentations is segmentations


def test_log_detections_prints_json_and_returns_input():
    stream = io.StringIO()
    detections = [{"box": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "class_id": 1}]

    result = LogDetections(
        model_path="model.onnx",
        image_path="image.jpg",
        annotated_image_path="image_model.jpg",
        stream=stream,
    )(detections)

    assert result is detections
    output = stream.getvalue()
    assert '"model": "model.onnx"' in output
    assert '"image": "image.jpg"' in output
    assert '"annotated_image": "image_model.jpg"' in output


def test_log_detections_at_one_prints_json_and_returns_input():
    stream = io.StringIO()
    payload = (
        "prefix",
        [{"box": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "class_id": 1}],
    )

    result = LogDetections(
        model_path="model.onnx",
        image_path="image.jpg",
        annotated_image_path="image_model.jpg",
        stream=stream,
        at=1,
    )(payload)

    assert result is payload
    output = stream.getvalue()
    assert '"model": "model.onnx"' in output
    assert '"image": "image.jpg"' in output
    assert '"annotated_image": "image_model.jpg"' in output
