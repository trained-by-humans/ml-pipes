import io
from pathlib import Path

import numpy as np
import pytest

from ml_pipes import Pipeline, PipelineValidationError
from ml_pipes.ops import (
    ArgMax,
    Cast,
    ConvertBoxFormat,
    DrawBoxes,
    Extract,
    FilterBy,
    GatherScores,
    Infer,
    LogDetections,
    MapToObjects,
    NMS,
    Normalize,
    ProjectBoxes,
    ProjectMasks,
    ProjectRoIMasks,
    ReconstructMasks,
    Resize,
    SaveImage,
    Sigmoid,
    Slice,
    Softmax,
    Squeeze,
    Pick,
    ToDetections,
    ToSegmentations,
    Transpose,
)
from ml_pipes.types import (
    Detections,
    ImagePayload,
    ResizeTransform,
    RuntimeOutputs,
    Segmentations,
    TensorPayload,
    TensorRegistry,
)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def test_cast_does_not_silence_downstream_type_check():
    pipeline = Pipeline([Cast("float32"), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_cast_establishes_tensorpayload_input_contract():
    contract = Pipeline([Cast("float32")]).validate()

    assert contract is not None
    assert contract.input_type == (TensorPayload | tuple[TensorPayload, ...])


class MakeTensor:
    def __call__(self, value: int) -> TensorPayload:
        return TensorPayload(array=np.array([value], dtype=np.float32), layout="N", dtype="float32")


class AcceptTensor:
    def __call__(self, value: TensorPayload) -> int:
        return int(value.array[0])


def test_cast_preserves_single_tensor_contract_for_typed_pipeline():
    contract = Pipeline([MakeTensor(), Cast("float16"), AcceptTensor()]).validate()

    assert contract is not None
    assert contract.input_type is int


# ---------------------------------------------------------------------------
# Pick
# ---------------------------------------------------------------------------

def test_pick_validation_propagates_element_type():
    pipeline = Pipeline([IntToPair(), Pick(0), IntToString()])

    pipeline.validate()


def test_pick_validation_rejects_wrong_downstream_type():
    pipeline = Pipeline([IntToPair(), Pick(0), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pick_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Pick(5)])

    with pytest.raises(PipelineValidationError, match="Pick\\(5\\) is out of bounds"):
        pipeline.validate()


def test_pick_out_of_bounds_silent_on_vague_input():
    pipeline = Pipeline([Pick(5)])

    pipeline.validate()  # must not raise


def test_pick_establishes_tuple_input_boundary_from_downstream_type():
    contract = Pipeline([Pick(0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is tuple

def test_resize_op_can_do_plain_resize_without_padding():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    resized, transform = Resize(target_size=(40, 40), mode="resize")(payload)

    assert resized.array.shape == (40, 40, 3)
    assert transform.scale == (2.0, 4.0)
    assert transform.pad == (0.0, 0.0)
    assert transform.resized_shape == (40, 40)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def test_extract_creates_registry_with_named_tensors():
    array = np.array([[1.0, 2.0]], dtype=np.float32)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=array, layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    registry = Extract("output_0")(outputs)

    assert np.array_equal(registry["output_0"], array)


def test_extract_renames_tensor_with_as_():
    array = np.array([[1.0, 2.0]], dtype=np.float32)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=array, layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    registry = Extract("output_0", as_="preds")(outputs)

    assert np.array_equal(registry["preds"], array)


def test_extract_raises_on_missing_output_name():
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=np.zeros((1,), dtype=np.float32), layout="UNKNOWN", dtype="float32"),),
        names=("output_0",),
    )

    with pytest.raises(KeyError, match="not found"):
        Extract("missing")(outputs)


# ---------------------------------------------------------------------------
# Squeeze / Transpose / Slice
# ---------------------------------------------------------------------------

def test_squeeze_removes_size_one_batch_dim():
    registry = TensorRegistry({"preds": np.zeros((1, 5, 10), dtype=np.float32)})

    result = Squeeze("preds")(registry)

    assert result["preds"].shape == (5, 10)


def test_transpose_swaps_axes():
    registry = TensorRegistry({"preds": np.zeros((5, 10), dtype=np.float32)})

    result = Transpose("preds")(registry)

    assert result["preds"].shape == (10, 5)


def test_slice_extracts_column_range():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    registry = TensorRegistry({"preds": data})

    result = Slice("preds", at=slice(None, 2), as_="boxes")(registry)

    assert result["boxes"].shape == (3, 2)
    assert np.array_equal(result["boxes"], data[:, :2])


# ---------------------------------------------------------------------------
# ArgMax / GatherScores
# ---------------------------------------------------------------------------

def test_argmax_picks_highest_score_index_per_row():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    registry = TensorRegistry({"scores": scores})

    result = ArgMax("scores", as_="classes")(registry)

    assert result["classes"].tolist() == [1, 0]


def test_gather_scores_picks_class_score():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    classes = np.array([1, 0], dtype=np.int32)
    registry = TensorRegistry({"scores": scores, "classes": classes})

    result = GatherScores("scores", "classes")(registry)

    assert np.allclose(result["scores"], [0.9, 0.8])


# ---------------------------------------------------------------------------
# Softmax / Sigmoid
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one_per_row():
    registry = TensorRegistry({"logits": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)})

    result = Softmax("logits")(registry)

    assert np.allclose(result["logits"].sum(axis=-1), [1.0])


def test_sigmoid_maps_zero_to_half():
    registry = TensorRegistry({"x": np.array([[0.0]], dtype=np.float32)})

    result = Sigmoid("x")(registry)

    assert np.allclose(result["x"], [[0.5]])


# ---------------------------------------------------------------------------
# ConvertBoxFormat
# ---------------------------------------------------------------------------

def test_convert_box_format_cxcywh_to_xyxy():
    # cx=10, cy=20, w=4, h=6  →  x1=8, y1=17, x2=12, y2=23
    registry = TensorRegistry({"boxes": np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)})

    result = ConvertBoxFormat(from_="cxcywh")(registry)

    assert np.allclose(result["boxes"], [[8.0, 17.0, 12.0, 23.0]])


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _make_registry(boxes, scores, classes):
    registry = TensorRegistry()
    registry["boxes"] = np.array(boxes, dtype=np.float32)
    registry["scores"] = np.array(scores, dtype=np.float32)
    registry["classes"] = np.array(classes, dtype=np.int32)
    return registry


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


# ---------------------------------------------------------------------------
# FilterBy
# ---------------------------------------------------------------------------

def test_filter_by_synchronises_extra_tensor_with_nms_kept_indices():
    registry = _make_registry(
        boxes=[[10, 10, 50, 50], [12, 12, 48, 48]],
        scores=[0.95, 0.85],
        classes=[0, 0],
    )
    registry["coefficients"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = NMS(kept_as="kept")(registry)
    result = FilterBy("coefficients", indices="kept")(result)

    assert result["coefficients"].shape == (1, 2)
    assert result["coefficients"].tolist() == [[1.0, 2.0]]


# ---------------------------------------------------------------------------
# ReconstructMasks
# ---------------------------------------------------------------------------

def test_reconstruct_masks_produces_correct_shape():
    coefficients = np.ones((2, 3), dtype=np.float32)    # N=2 detections, C=3 mask channels
    prototypes = np.ones((3, 4, 4), dtype=np.float32)   # C=3, H=4, W=4
    registry = TensorRegistry({"coefficients": coefficients, "prototypes": prototypes})

    result = ReconstructMasks("coefficients", "prototypes", as_="masks")(registry)

    assert result["masks"].shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# ProjectBoxes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ProjectMasks
# ---------------------------------------------------------------------------

def test_project_masks_produces_binary_masks_at_original_size():
    coefficients = np.array([[1.0]], dtype=np.float32)
    prototypes = np.ones((1, 4, 4), dtype=np.float32)
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(0.0, 0.0),
        original_shape=(2, 2),
        resized_shape=(4, 4),
    )
    registry = TensorRegistry()
    registry["boxes"] = np.array([[0.5, 0.5, 1.5, 1.5]], dtype=np.float32)
    registry["scores"] = np.array([0.9], dtype=np.float32)
    registry["classes"] = np.array([1], dtype=np.int32)
    registry["masks"] = (coefficients @ prototypes.reshape(1, -1)).reshape(1, 4, 4)

    result = ProjectMasks(mask_threshold=0.0)(registry, transform)

    assert len(result["masks"]) == 1
    assert result["masks"][0].shape == (2, 2)


# ---------------------------------------------------------------------------
# ToDetections / ToSegmentations
# ---------------------------------------------------------------------------

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


def test_to_segmentations_converts_registry_to_segmentations():
    registry = _make_registry(
        boxes=[[1.0, 2.0, 3.0, 4.0]],
        scores=[0.9],
        classes=[1],
    )
    registry["masks"] = np.zeros((1, 4, 4), dtype=np.uint8)

    result = ToSegmentations()(registry)

    assert isinstance(result, Segmentations)
    assert result.boxes == [[1.0, 2.0, 3.0, 4.0]]
    assert len(result.masks) == 1


# ---------------------------------------------------------------------------
# Cast
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Infer (mocked)
# ---------------------------------------------------------------------------

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
    with pytest.raises(ValueError, match="model dtype"):
        infer(value)


# ---------------------------------------------------------------------------
# DrawBoxes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SaveImage
# ---------------------------------------------------------------------------

def test_save_image_writes_output(tmp_path: Path):
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "annotated.jpg"

    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
    result = SaveImage(output_path)(payload)

    assert output_path.is_file()
    assert result is payload


# ---------------------------------------------------------------------------
# MapToObjects / LogDetections
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ProjectRoIMasks
# ---------------------------------------------------------------------------

def test_project_roi_masks_embeds_mask_into_canvas():
    transform = ResizeTransform(scale=(1.0, 1.0), pad=(0.0, 0.0), original_shape=(8, 8), resized_shape=(8, 8))
    registry = TensorRegistry()
    registry["boxes"] = np.array([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32)
    registry["masks"] = np.ones((1, 4, 4), dtype=np.float32)

    result = ProjectRoIMasks(mask_threshold=0.5)(registry, transform)

    canvas = result["masks"]
    assert canvas.shape == (1, 8, 8)
    assert np.all(canvas[0, 2:6, 2:6])
    assert not np.any(canvas[0, :2, :])


def test_project_roi_masks_preserves_fractional_box():
    # A box like [1.1, 1.1, 1.9, 1.9] truncated with astype(int) collapses to
    # width=0, height=0 and the mask is silently dropped.  floor/ceil must be
    # used so the ROI expands to [1, 1, 2, 2] and the mask is preserved.
    transform = ResizeTransform(scale=(1.0, 1.0), pad=(0.0, 0.0), original_shape=(4, 4), resized_shape=(4, 4))
    registry = TensorRegistry()
    registry["boxes"] = np.array([[1.1, 1.1, 1.9, 1.9]], dtype=np.float32)
    registry["masks"] = np.ones((1, 1, 1), dtype=np.float32)

    result = ProjectRoIMasks(mask_threshold=0.5)(registry, transform)

    assert np.any(result["masks"][0]), "mask was silently dropped due to truncation"
