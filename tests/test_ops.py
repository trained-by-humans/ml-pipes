from pathlib import Path

import numpy as np

from ml_pipes.ops import (
    DecodePredictionsOp,
    DrawBoxesOp,
    NMSOp,
    ProjectToInputOp,
    SaveImageOp,
)
from ml_pipes.transforms import ResizeTransform
from ml_pipes.types import DetectionBatch, DetectionResult, ImagePayload, TensorPayload


def test_decode_predictions_accepts_channel_first_yolov8_output():
    raw = np.array(
        [
            [
                [10.0, 20.0],
                [12.0, 22.0],
                [4.0, 8.0],
                [6.0, 10.0],
                [0.9, 0.1],
                [0.05, 0.8],
            ]
        ],
        dtype=np.float32,
    )

    result = DecodePredictionsOp()(TensorPayload(array=raw, layout="UNKNOWN", dtype="float32"))

    assert result.boxes.shape == (2, 4)
    assert result.classes.tolist() == [0, 1]
    assert np.allclose(result.scores, [0.9, 0.8])


def test_nms_keeps_overlapping_boxes_from_different_classes():
    detections = DetectionBatch(
        boxes=np.array(
            [
                [10, 10, 50, 50],
                [12, 12, 48, 48],
            ],
            dtype=np.float32,
        ),
        scores=np.array([0.95, 0.9], dtype=np.float32),
        classes=np.array([0, 1], dtype=np.int32),
    )

    result = NMSOp(conf_threshold=0.25, iou_threshold=0.4)(detections)

    assert result.boxes.shape == (2, 4)
    assert result.classes.tolist() == [0, 1]


def test_nms_suppresses_same_class_overlap():
    detections = DetectionBatch(
        boxes=np.array(
            [
                [10, 10, 50, 50],
                [12, 12, 48, 48],
                [100, 100, 140, 140],
            ],
            dtype=np.float32,
        ),
        scores=np.array([0.95, 0.85, 0.8], dtype=np.float32),
        classes=np.array([0, 0, 0], dtype=np.int32),
    )

    result = NMSOp(conf_threshold=0.25, iou_threshold=0.4)(detections)

    assert result.boxes.shape == (2, 4)
    assert np.allclose(result.scores, [0.95, 0.8])


def test_project_to_input_reverses_padding_and_scale():
    transform = ResizeTransform(scale=2.0, pad=(10.0, 20.0), original_shape=(100, 200))
    detections = DetectionBatch(
        boxes=np.array([[30.0, 40.0, 110.0, 120.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([3], dtype=np.int32),
    )
    result = ProjectToInputOp()(detections, transform)

    assert result.boxes == [[10.0, 10.0, 50.0, 50.0]]
    assert result.scores == [0.8999999761581421]
    assert result.classes == [3]


def test_project_to_input_clips_boxes_to_original_bounds():
    transform = ResizeTransform(scale=2.0, pad=(10.0, 20.0), original_shape=(100, 200))
    detections = DetectionBatch(
        boxes=np.array([[-50.0, -50.0, 500.0, 400.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.int32),
    )
    result = ProjectToInputOp()(detections, transform)

    assert result.boxes == [[0.0, 0.0, 200.0, 100.0]]


def test_draw_boxes_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = DetectionResult(
        boxes=[[4.0, 4.0, 20.0, 20.0]],
        scores=[0.9],
        classes=[1],
    )
    result = DrawBoxesOp(class_names=["zero", "one"], color=(0, 255, 0))(detections, image)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)


def test_save_image_writes_output(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "annotated.jpg"

    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
    result = SaveImageOp(output_path)(payload)

    assert output_path.is_file()
    assert result is payload
