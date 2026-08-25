from __future__ import annotations

import io

import numpy as np
import pytest

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import (
    ConvertBoxFormat,
    DrawBoxes,
    FilterTensorsByBoxArea,
    FilterTensorsByClasses,
    FilterTensorsByScore,
    ImagePayload,
    LogDetections,
    NMM,
    NMS,
    ProjectBoxes,
    ResizeTransform,
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


def test_filter_tensors_by_box_area_filters_aligned_registry_values():
    registry = _make_registry(
        boxes=[[0, 0, 1, 1], [0, 0, 4, 4]],
        scores=[0.4, 0.9],
        classes=[0, 1],
    )

    result = FilterTensorsByBoxArea("scores", "classes", min_area=4)(registry)

    assert result["boxes"].tolist() == [[0.0, 0.0, 4.0, 4.0]]
    assert np.allclose(result["scores"], [0.9])


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


def test_nmm_uses_uniform_weights_when_overlapping_scores_are_zero():
    registry = _make_registry(
        boxes=[[0, 0, 2, 2], [2, 2, 4, 4]],
        scores=[0.0, 0.0],
        classes=[1, 1],
    )

    result = NMM(iou_threshold=0.0)(registry)

    assert np.allclose(result["boxes"], [[1.0, 1.0, 3.0, 3.0]])
    assert result["boxes"].dtype == np.float32
    assert result["scores"].dtype == np.float32
    assert result["classes"].dtype == np.int32


def test_nms_requires_classes_tensor():
    with pytest.raises(ValueError, match="requires a classes tensor"):
        NMS(classes=None)


def test_draw_boxes_preserves_rgb_metadata_and_translates_bgr_color():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    registry = _make_registry([[0, 0, 3, 3]], [0.9], [1])

    result, _ = DrawBoxes(color=(1, 2, 3), thickness=1)(
        ImagePayload(array=image, color_space="RGB", layout="HWC"), registry
    )

    assert result.color_space == "RGB"
    assert result.layout == "HWC"
    assert result.array[0, 0].tolist() == [3, 2, 1]


def test_draw_boxes_accepts_class_agnostic_detections():
    registry = TensorRegistry(
        {
            "boxes": np.array([[0, 0, 3, 3]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
        }
    )

    result, returned_registry = DrawBoxes(classes=None)(
        ImagePayload(array=np.zeros((4, 4, 3), dtype=np.uint8), color_space="BGR", layout="HWC"),
        registry,
    )

    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_draw_boxes_requires_classes_when_class_names_are_configured():
    with pytest.raises(ValueError, match="class_names requires a classes tensor"):
        DrawBoxes(classes=None, class_names=["person"])


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


def test_draw_boxes_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ImagePayload(array=image, color_space="BGR", layout="HWC")
    registry = _make_registry([[4.0, 4.0, 20.0, 20.0]], [0.9], [1])

    result, returned_registry = DrawBoxes(class_names=["zero", "one"], color=(0, 255, 0))(source, registry)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_draw_boxes_accepts_source_names_before_rendering_options():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    registry = TensorRegistry({
        "candidate_boxes": np.array([[4, 4, 20, 20]], dtype=np.float32),
        "candidate_scores": np.array([0.9], dtype=np.float32),
        "candidate_classes": np.array([1], dtype=np.int32),
    })

    result, returned_registry = DrawBoxes("candidate_boxes", "candidate_scores", "candidate_classes", class_names=["zero", "one"])(
        ImagePayload(array=image, color_space="BGR", layout="HWC"), registry
    )

    assert np.any(result.array != 0)
    assert returned_registry is registry


def test_log_detections_prints_json_and_returns_input():
    stream = io.StringIO()
    registry = _make_registry([[1.0, 2.0, 3.0, 4.0]], [0.9], [1])

    result = LogDetections(
        model_path="model.onnx",
        image_path="image.jpg",
        annotated_image_path="image_model.jpg",
        stream=stream,
    )(registry)

    assert result is registry
    output = stream.getvalue()
    assert '"model": "model.onnx"' in output
    assert '"image": "image.jpg"' in output
    assert '"annotated_image": "image_model.jpg"' in output


def test_log_detections_at_one_prints_json_and_returns_input():
    stream = io.StringIO()
    payload = ("prefix", _make_registry([[1.0, 2.0, 3.0, 4.0]], [0.9], [1]))

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
