import io
from pathlib import Path

import numpy as np

from ml_pipes.ops import (
    Cast,
    DecodePredictionsOp,
    DecodeSegmentationOp,
    DrawBoxes,
    Infer,
    LogDetections,
    MapToObjects,
    NMSOp,
    Normalize,
    ProjectSegmentationsOp,
    ProjectToInputOp,
    Resize,
    SaveImage,
    SegmentationNMSOp,
)
from ml_pipes.transforms import ResizeTransform
from ml_pipes.types import (
    DetectionArrays,
    Detections,
    ImagePayload,
    RuntimeOutputs,
    SegmentationCandidates,
    TensorPayload,
)


def test_resize_op_can_do_plain_resize_without_padding():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    resized, transform = Resize(target_size=(40, 40), mode="resize")(payload)

    assert resized.array.shape == (40, 40, 3)
    assert transform.scale == (2.0, 4.0)
    assert transform.pad == (0.0, 0.0)
    assert transform.resized_shape == (40, 40)


def test_normalize_op_can_keep_bgr_and_hwc_without_batch():
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    tensor = Normalize(
        output_color_space="BGR",
        output_layout="HWC",
        add_batch_dim=False,
        scale=1.0,
    )(payload)

    assert tensor.layout == "HWC"
    assert tensor.dtype == "float32"
    assert tensor.array.shape == (1, 1, 3)
    assert tensor.array.tolist() == [[[10.0, 20.0, 30.0]]]


def test_normalize_op_preserves_floating_input_dtype():
    image = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float16)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    tensor = Normalize(
        output_color_space="BGR",
        output_layout="HWC",
        add_batch_dim=False,
        scale=1.0,
    )(payload)

    assert tensor.array.dtype == np.float16
    assert tensor.dtype == "float16"


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

    result = DecodePredictionsOp()(
        RuntimeOutputs(
            tensors=(TensorPayload(array=raw, layout="UNKNOWN", dtype="float32"),),
            names=("output_0",),
        )
    )

    assert result.boxes.shape == (2, 4)
    assert result.classes.tolist() == [0, 1]
    assert np.allclose(result.scores, [0.9, 0.8])


def test_decode_predictions_can_read_xyxy_without_transpose():
    raw = np.array(
        [
            [1.0, 2.0, 11.0, 12.0, 0.1, 0.9],
            [5.0, 6.0, 15.0, 16.0, 0.8, 0.2],
        ],
        dtype=np.float32,
    )

    result = DecodePredictionsOp(
        input_box_format="xyxy",
        transpose_output="never",
        class_start_index=4,
    )(
        RuntimeOutputs(
            tensors=(TensorPayload(array=raw, layout="UNKNOWN", dtype="float32"),),
            names=("output_0",),
        )
    )

    assert result.boxes.tolist() == [[1.0, 2.0, 11.0, 12.0], [5.0, 6.0, 15.0, 16.0]]
    assert result.classes.tolist() == [1, 0]


def test_decode_predictions_can_apply_sigmoid_to_scores():
    raw = np.array(
        [
            [
                [10.0],
                [20.0],
                [4.0],
                [6.0],
                [0.0],
                [2.0],
            ]
        ],
        dtype=np.float32,
    )

    result = DecodePredictionsOp(score_activation="sigmoid")(
        RuntimeOutputs(
            tensors=(TensorPayload(array=raw, layout="UNKNOWN", dtype="float32"),),
            names=("output_0",),
        )
    )

    assert result.classes.tolist() == [1]
    assert np.allclose(result.scores, [1.0 / (1.0 + np.exp(-2.0))])


def test_decode_predictions_can_select_export_output_by_index():
    aux = TensorPayload(array=np.zeros((1, 2), dtype=np.float32), layout="UNKNOWN", dtype="float32")
    predictions = TensorPayload(
        array=np.array(
            [
                [
                    [10.0],
                    [20.0],
                    [4.0],
                    [6.0],
                    [0.9],
                    [0.1],
                ]
            ],
            dtype=np.float32,
        ),
        layout="UNKNOWN",
        dtype="float32",
    )
    runtime_outputs = RuntimeOutputs(tensors=(aux, predictions), names=("aux", "predictions"))

    result = DecodePredictionsOp(export_output_index=1)(runtime_outputs)

    assert result.boxes.shape == (1, 4)
    assert result.classes.tolist() == [0]


def test_decode_predictions_can_select_export_output_by_name():
    aux = TensorPayload(array=np.zeros((1, 2), dtype=np.float32), layout="UNKNOWN", dtype="float32")
    predictions = TensorPayload(
        array=np.array(
            [
                [1.0, 2.0, 11.0, 12.0, 0.1, 0.9],
                [5.0, 6.0, 15.0, 16.0, 0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        layout="UNKNOWN",
        dtype="float32",
    )
    runtime_outputs = RuntimeOutputs(tensors=(aux, predictions), names=("aux", "pred_boxes"))

    result = DecodePredictionsOp(
        export_output_name="pred_boxes",
        input_box_format="xyxy",
        transpose_output="never",
    )(runtime_outputs)

    assert result.boxes.tolist() == [[1.0, 2.0, 11.0, 12.0], [5.0, 6.0, 15.0, 16.0]]
    assert result.classes.tolist() == [1, 0]


def test_decode_segmentation_can_select_detection_and_prototype_outputs():
    detections = TensorPayload(
        array=np.array(
            [
                [
                    [10.0],
                    [20.0],
                    [4.0],
                    [6.0],
                    [0.9],
                    [0.1],
                    [0.2],
                    [0.8],
                ]
            ],
            dtype=np.float32,
        ),
        layout="UNKNOWN",
        dtype="float32",
    )
    prototypes = TensorPayload(
        array=np.ones((1, 2, 4, 4), dtype=np.float32),
        layout="UNKNOWN",
        dtype="float32",
    )
    runtime_outputs = RuntimeOutputs(
        tensors=(prototypes, detections),
        names=("proto", "pred"),
    )

    result = DecodeSegmentationOp(
        export_detection_output_name="pred",
        export_prototype_output_name="proto",
        num_masks=2,
    )(runtime_outputs)

    assert result.boxes.shape == (1, 4)
    assert result.mask_coefficients.shape == (1, 2)
    assert result.prototypes.shape == (2, 4, 4)
    assert result.classes.tolist() == [0]


def test_segmentation_nms_preserves_mask_coefficients():
    candidates = SegmentationCandidates(
        boxes=np.array([[10, 10, 40, 40], [12, 12, 38, 38]], dtype=np.float32),
        scores=np.array([0.95, 0.85], dtype=np.float32),
        classes=np.array([0, 0], dtype=np.int32),
        mask_coefficients=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        prototypes=np.ones((2, 8, 8), dtype=np.float32),
    )

    result = SegmentationNMSOp(conf_threshold=0.25, iou_threshold=0.4)(candidates)

    assert result.boxes.shape == (1, 4)
    assert result.mask_coefficients.tolist() == [[1.0, 2.0]]


def test_project_segmentations_projects_masks_to_original_image():
    candidates = SegmentationCandidates(
        boxes=np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.int32),
        mask_coefficients=np.array([[1.0]], dtype=np.float32),
        prototypes=np.ones((1, 4, 4), dtype=np.float32),
    )
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(0.0, 0.0),
        original_shape=(2, 2),
        resized_shape=(4, 4),
    )

    result = ProjectSegmentationsOp(mask_threshold=0.0)(candidates, transform)

    assert result.boxes == [[0.5, 0.5, 1.5, 1.5]]
    assert result.scores == [0.8999999761581421]
    assert result.classes == [1]
    assert len(result.masks) == 1
    assert result.masks[0].shape == (2, 2)
    assert np.all(result.masks[0] == 1)


def test_cast_tensor_op_can_cast_iterable_of_tensor_payloads():
    tensors = (
        TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        TensorPayload(array=np.array([[3.0, 4.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
    )

    result = Cast("float32")(tensors)

    assert isinstance(result, tuple)
    assert result[0].array.dtype == np.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == np.float32
    assert result[1].dtype == "float32"


def test_cast_tensor_op_can_cast_selected_dataclass_field():
    runtime_outputs = RuntimeOutputs(
        tensors=(
            TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        ),
        names=("output_0",),
    )

    result = Cast("float32", field="tensors")(runtime_outputs)

    assert isinstance(result, RuntimeOutputs)
    assert result.names == ("output_0",)
    assert result.tensors[0].array.dtype == np.float32
    assert result.tensors[0].dtype == "float32"


def test_cast_tensor_op_can_cast_single_tensor_payload():
    payload = TensorPayload(
        array=np.array([[1.0, 2.0]], dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = Cast("float16")(payload)

    assert result.array.dtype == np.float16
    assert result.dtype == "float16"


def test_infer_op_requires_requested_model_dtype():
    class _FakeSession:
        def run(self, _output_names, _inputs):
            return [np.array([[1.0, 2.0]], dtype=np.float16)]

    infer = Infer.__new__(Infer)
    infer.session = _FakeSession()
    infer.input_name = "images"
    infer.input_layout = "NCHW"
    infer.model_dtype = np.dtype("float32")
    infer.output_layouts = ("UNKNOWN",)
    infer.output_names = ("output_0",)

    value = TensorPayload(
        array=np.zeros((1, 3, 8, 8), dtype=np.float16),
        layout="NCHW",
        dtype="float16",
    )
    try:
        infer(value)
    except ValueError as error:
        assert "model dtype" in str(error)
    else:
        raise AssertionError("Infer should reject mismatched model dtype")


def test_nms_keeps_overlapping_boxes_from_different_classes():
    detections = DetectionArrays(
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
    detections = DetectionArrays(
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
    transform = ResizeTransform(scale=(2.0, 2.0), pad=(10.0, 20.0), original_shape=(100, 200), resized_shape=(240, 420))
    detections = DetectionArrays(
        boxes=np.array([[30.0, 40.0, 110.0, 120.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([3], dtype=np.int32),
    )
    result = ProjectToInputOp()(detections, transform)

    assert result.boxes == [[10.0, 10.0, 50.0, 50.0]]
    assert result.scores == [0.8999999761581421]
    assert result.classes == [3]


def test_project_to_input_clips_boxes_to_original_bounds():
    transform = ResizeTransform(scale=(2.0, 2.0), pad=(10.0, 20.0), original_shape=(100, 200), resized_shape=(240, 420))
    detections = DetectionArrays(
        boxes=np.array([[-50.0, -50.0, 500.0, 400.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([1], dtype=np.int32),
    )
    result = ProjectToInputOp()(detections, transform)

    assert result.boxes == [[0.0, 0.0, 200.0, 100.0]]


def test_draw_boxes_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = Detections(
        boxes=[[4.0, 4.0, 20.0, 20.0]],
        scores=[0.9],
        classes=[1],
    )
    result = DrawBoxes(class_names=["zero", "one"], color=(0, 255, 0))(detections, image)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)


def test_save_image_writes_output(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "annotated.jpg"

    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
    result = SaveImage(output_path)(payload)

    assert output_path.is_file()
    assert result is payload


def test_map_to_objects_can_convert_detection_result():
    detections = Detections(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )

    result = MapToObjects(
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
