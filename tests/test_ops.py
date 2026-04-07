from pathlib import Path

import numpy as np

from ml_pipes.core import Context, Value
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

    result = DecodePredictionsOp()(Value(TensorPayload(array=raw, layout="UNKNOWN", dtype="float32")))

    assert result.data.boxes.shape == (2, 4)
    assert result.data.classes.tolist() == [0, 1]
    assert np.allclose(result.data.scores, [0.9, 0.8])


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

    result = NMSOp(conf_threshold=0.25, iou_threshold=0.4)(Value(detections))

    assert result.data.boxes.shape == (2, 4)
    assert result.data.classes.tolist() == [0, 1]


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

    result = NMSOp(conf_threshold=0.25, iou_threshold=0.4)(Value(detections))

    assert result.data.boxes.shape == (2, 4)
    assert np.allclose(result.data.scores, [0.95, 0.8])


def test_project_to_input_reverses_padding_and_scale():
    transform = ResizeTransform(scale=2.0, pad=(10.0, 20.0), original_shape=(100, 200))
    detections = DetectionBatch(
        boxes=np.array([[30.0, 40.0, 110.0, 120.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([3], dtype=np.int32),
    )
    value = Value(detections, Context((transform,)))

    result = ProjectToInputOp()(value)

    assert result.data.boxes == [[10.0, 10.0, 50.0, 50.0]]
    assert result.data.scores == [0.8999999761581421]
    assert result.data.classes == [3]


def test_project_to_input_clips_boxes_to_original_bounds():
    transform = ResizeTransform(scale=2.0, pad=(10.0, 20.0), original_shape=(100, 200))
    detections = DetectionBatch(
        boxes=np.array([[-50.0, -50.0, 500.0, 400.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.int32),
    )
    value = Value(detections, Context((transform,)))

    result = ProjectToInputOp()(value)

    assert result.data.boxes == [[0.0, 0.0, 200.0, 100.0]]


def test_draw_boxes_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = DetectionResult(
        boxes=[[4.0, 4.0, 20.0, 20.0]],
        scores=[0.9],
        classes=[1],
    )
    context = Context(metadata={"source_image": image})

    result = DrawBoxesOp(class_names=["zero", "one"], color=(0, 255, 0))(Value(detections, context))

    assert result.data.array.shape == image.shape
    assert result.data.color_space == "BGR"
    assert result.data.layout == "HWC"
    assert np.any(result.data.array != 0)


def test_save_image_writes_output(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    drawn = Value(
        data=ImagePayload(array=image, color_space="BGR", layout="HWC"),
        context=Context(),
    )
    output_path = tmp_path / "annotated.jpg"

    result = SaveImageOp(output_path)(drawn)

    assert output_path.is_file()
    assert result is drawn
