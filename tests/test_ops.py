import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_pipes import Pipeline, PipelineValidationError
from ml_pipes.ops import (
    ApplyTensorMask,
    AsType,
    ArgMax,
    BinarizeTensor,
    BinarizeTensorByThreshold,
    ConvertColorSpace,
    CreateTensorMask,
    CreateTensorMaskByThreshold,
    ConvertBoxFormat,
    DrawBoxes,
    DrawMasks,
    Extract,
    FilterTensorsByClasses,
    FilterTensorsByMasksArea,
    FilterPredictions,
    FilterPredictionsByArea,
    FilterPredictionsByClass,
    FilterPredictionsByScore,
    FilterTensors,
    FilterTensorsByScore,
    GatherScores,
    Infer,
    LogDetections,
    MapPredictionsToObjects,
    MultiplyTensors,
    MeanMaskScores,
    MasksToBoxes,
    NMS,
    Normalize,
    ProjectBoxes,
    ProjectMasks,
    ProjectRoIMasks,
    ReconstructMasks,
    Resize,
    ResizeMasks,
    SaveImage,
    Select,
    SelectTensors,
    Sigmoid,
    Slice,
    Softmax,
    SortTensorsBy,
    TopK,
    TopKIndices2D,
    Squeeze,
    Pick,
    ToDetections,
    ToSegmentations,
    Transpose,
    WeightMasksByScores,
)
from ml_pipes.types import (
    Detections,
    ImagePayload,
    Prediction,
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


class MakeImage:
    def __call__(self, value: int) -> ImagePayload:
        return ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="BGR", layout="HWC")


class AcceptArray:
    def __call__(self, value: np.ndarray) -> int:
        return int(value.shape[0])


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def test_as_type_does_not_silence_downstream_type_check():
    pipeline = Pipeline([AsType("float32"), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_as_type_establishes_tensor_like_input_contract():
    contract = Pipeline([AsType("float32")]).validate()

    assert contract is not None
    assert contract.input_type == (
        TensorPayload
        | np.ndarray
        | tuple[TensorPayload, ...]
        | tuple[np.ndarray, ...]
        | list[TensorPayload]
        | list[np.ndarray]
    )


class MakeTensor:
    def __call__(self, value: int) -> TensorPayload:
        return TensorPayload(array=np.array([value], dtype=np.float32), layout="N", dtype="float32")


class AcceptTensor:
    def __call__(self, value: TensorPayload) -> int:
        return int(value.array[0])


def test_as_type_preserves_single_tensor_contract_for_typed_pipeline():
    contract = Pipeline([MakeTensor(), AsType("float16"), AcceptTensor()]).validate()

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


def test_pick_negative_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Pick(-3)])

    with pytest.raises(PipelineValidationError, match="Pick\\(-3\\) is out of bounds"):
        pipeline.validate()


def test_pick_out_of_bounds_silent_on_vague_input():
    pipeline = Pipeline([Pick(5)])

    pipeline.validate()  # must not raise


def test_pick_validation_rejects_known_non_tuple_input():
    pipeline = Pipeline([IntToString(), Pick(0)])

    with pytest.raises(PipelineValidationError, match="Pick requires a tuple boundary"):
        pipeline.validate()


def test_pick_establishes_tuple_input_boundary_from_downstream_type():
    contract = Pipeline([Pick(0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type == tuple[Any, ...]


def test_select_validation_propagates_array_attribute_type():
    contract = Pipeline([MakeImage(), Select("array"), AcceptArray()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_propagates_nested_attribute_type():
    contract = Pipeline([MakeImage(), Select(("spatial_shape", 0)), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_accepts_dotted_string_selector():
    contract = Pipeline([MakeImage(), Select("spatial_shape.0"), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_accepts_variadic_selector_parts():
    contract = Pipeline([MakeImage(), Select("spatial_shape", 0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_tuple_index_rejects_known_non_tuple_input():
    pipeline = Pipeline([IntToString(), Select(0)])

    with pytest.raises(PipelineValidationError, match="indexable"):
        pipeline.validate()


def test_select_tuple_index_establishes_tuple_input_boundary():
    contract = Pipeline([Select(0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is Any

def test_resize_op_can_do_plain_resize_without_padding():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    resized, transform = Resize(target_size=(40, 40), mode="resize")(payload)

    assert resized.array.shape == (40, 40, 3)
    assert transform.scale == (2.0, 4.0)
    assert transform.pad == (0.0, 0.0)
    assert transform.resized_shape == (40, 40)


def test_image_payload_exposes_derived_shape_properties():
    payload = ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="BGR", layout="HWC")

    assert payload.shape == (10, 20, 3)
    assert payload.spatial_shape == (10, 20)
    assert payload.height == 10
    assert payload.width == 20
    assert payload.size == (20, 10)
    assert payload.dtype == "uint8"
    assert payload.ndim == 3
    assert payload.channels == 3


def test_image_payload_spatial_shape_uses_layout_for_chw():
    payload = ImagePayload(array=np.zeros((3, 10, 20), dtype=np.uint8), color_space="BGR", layout="CHW")

    assert payload.shape == (3, 10, 20)
    assert payload.spatial_shape == (10, 20)
    assert payload.height == 10
    assert payload.width == 20
    assert payload.size == (20, 10)
    assert payload.channels == 3


def test_convert_color_space_converts_bgr_to_rgb_and_preserves_metadata():
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    payload = ImagePayload(array=image, color_space="BGR", layout="HWC")

    converted = ConvertColorSpace("RGB")(payload)

    assert converted.color_space == "RGB"
    assert converted.layout == "HWC"
    assert converted.dtype == "uint8"
    assert converted.array.flags.c_contiguous
    assert converted.array.tolist() == [[[30, 20, 10], [60, 50, 40]]]


def test_convert_color_space_uses_channel_axis_from_layout():
    image = np.array(
        [
            [[10, 40]],
            [[20, 50]],
            [[30, 60]],
        ],
        dtype=np.uint8,
    )
    payload = ImagePayload(array=image, color_space="BGR", layout="CHW")

    converted = ConvertColorSpace("RGB")(payload)

    assert converted.layout == "CHW"
    assert converted.color_space == "RGB"
    assert converted.array.flags.c_contiguous
    assert converted.array.tolist() == [[[30, 60]], [[20, 50]], [[10, 40]]]


def test_convert_color_space_rejects_non_three_channel_input():
    payload = ImagePayload(array=np.zeros((10, 20, 1), dtype=np.uint8), color_space="BGR", layout="HWC")

    with pytest.raises(ValueError, match="3-channel"):
        ConvertColorSpace("RGB")(payload)


def test_convert_color_space_rejects_unknown_source_color_space():
    payload = ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="HSV", layout="HWC")

    with pytest.raises(ValueError, match="BGR/RGB input"):
        ConvertColorSpace("RGB")(payload)


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


def test_argmax_axis_zero_handles_empty_leading_dimension():
    registry = TensorRegistry({"scores": np.zeros((0, 2, 3), dtype=np.float32)})

    result = ArgMax("scores", axis=0, as_="classes")(registry)

    assert result["classes"].dtype == np.int32
    assert result["classes"].shape == (2, 3)
    assert np.array_equal(result["classes"], np.zeros((2, 3), dtype=np.int32))


def test_gather_scores_picks_class_score():
    scores = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    classes = np.array([1, 0], dtype=np.int32)
    registry = TensorRegistry({"scores": scores, "classes": classes})

    result = GatherScores("scores", "classes")(registry)

    assert np.allclose(result["scores"], [0.9, 0.8])


def test_topk_picks_highest_values_and_indices():
    registry = TensorRegistry({"scores": np.array([0.1, 0.7, 0.2, 0.8], dtype=np.float32)})

    result = TopK("scores", k=3, values_as="top_scores", indices_as="top_indices")(registry)

    assert np.allclose(result["top_scores"], [0.8, 0.7, 0.2])
    assert result["top_indices"].tolist() == [3, 1, 2]


def test_topk_indices_2d_returns_values_and_row_col_indices():
    registry = TensorRegistry(
        {
            "class_probs": np.array(
                [
                    [0.1, 0.7, 0.2],
                    [0.8, 0.3, 0.6],
                ],
                dtype=np.float32,
            )
        }
    )

    result = TopKIndices2D(
        "class_probs",
        k=3,
        values_as="top_scores",
        row_indices_as="query_indices",
        col_indices_as="class_ids",
    )(registry)

    assert np.allclose(result["top_scores"], [0.8, 0.7, 0.6])
    assert result["query_indices"].tolist() == [1, 0, 1]
    assert result["class_ids"].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Softmax / Sigmoid
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one_per_row():
    registry = TensorRegistry({"logits": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)})

    result = Softmax("logits")(registry)

    assert np.allclose(result["logits"].sum(axis=-1), [1.0])


def test_softmax_handles_empty_reduction_axis():
    registry = TensorRegistry({"logits": np.zeros((0, 2, 3), dtype=np.float32)})

    result = Softmax("logits", axis=0)(registry)

    assert result["logits"].shape == (0, 2, 3)
    assert result["logits"].dtype == np.float32


def test_sigmoid_maps_zero_to_half():
    registry = TensorRegistry({"x": np.array([[0.0]], dtype=np.float32)})

    result = Sigmoid("x")(registry)

    assert np.allclose(result["x"], [[0.5]])


def test_sigmoid_is_stable_for_large_magnitude_values():
    registry = TensorRegistry({"x": np.array([[-1000.0, 1000.0]], dtype=np.float32)})

    result = Sigmoid("x")(registry)

    assert np.isfinite(result["x"]).all()
    assert result["x"].dtype == np.float32
    assert np.allclose(result["x"], [[0.0, 1.0]])


def test_multiply_tensors_uses_numpy_broadcasting():
    registry = TensorRegistry(
        {
            "left": np.array([[1.0], [2.0]], dtype=np.float32),
            "right": np.array([[10.0, 20.0]], dtype=np.float32),
        }
    )

    result = MultiplyTensors("left", "right", as_="product")(registry)

    assert np.allclose(result["product"], [[10.0, 20.0], [20.0, 40.0]])


def test_weight_masks_by_scores_broadcasts_scores_over_masks():
    registry = TensorRegistry(
        {
            "scores": np.array([0.5, 2.0], dtype=np.float32),
            "masks": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=np.float32,
            ),
        }
    )

    result = WeightMasksByScores(as_="weighted_masks")(registry)

    assert np.allclose(
        result["weighted_masks"],
        [
            [[0.5, 1.0], [1.5, 2.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ],
    )


def test_resize_masks_to_image_resizes_mask_stack():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                ],
                dtype=np.float32,
            )
        }
    )

    result = ResizeMasks(masks="masks", as_="resized_masks")(registry, (4, 6))

    assert result["resized_masks"].shape == (1, 4, 6)
    assert result["resized_masks"].dtype == np.float32


def test_mean_mask_scores_computes_mean_over_binary_support():
    registry = TensorRegistry(
        {
            "selected_masks": np.array(
                [
                    [[0.0, 1.0], [0.5, 0.0]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ],
                dtype=np.float32,
            ),
            "binary_masks": np.array(
                [
                    [[False, True], [True, False]],
                    [[True, False], [False, True]],
                ]
            ),
        }
    )

    result = MeanMaskScores(masks="selected_masks", as_="mean_mask_scores")(registry)

    assert np.allclose(result["mean_mask_scores"], [0.75, 0.5])


def test_mean_mask_scores_handles_empty_masks():
    registry = TensorRegistry(
        {
            "selected_masks": np.zeros((0, 2, 2), dtype=np.float32),
            "binary_masks": np.zeros((0, 2, 2), dtype=bool),
        }
    )

    result = MeanMaskScores(masks="selected_masks", as_="mean_mask_scores")(registry)

    assert result["mean_mask_scores"].shape == (0,)
    assert result["mean_mask_scores"].dtype == np.float64


def test_mean_mask_scores_handles_empty_masks_without_binary_masks():
    registry = TensorRegistry({"selected_masks": np.zeros((0, 2, 2), dtype=np.float32)})

    result = MeanMaskScores(masks="selected_masks", binary_masks=None, as_="mean_mask_scores")(registry)

    assert result["mean_mask_scores"].shape == (0,)
    assert result["mean_mask_scores"].dtype == np.float32


def test_masks_to_boxes_converts_masks_to_xyxy():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[False, True, True], [False, True, False]],
                    [[False, False, False], [False, False, False]],
                ]
            )
        }
    )

    result = MasksToBoxes(as_="boxes")(registry)

    assert np.allclose(result["boxes"][0], [1.0, 0.0, 3.0, 2.0])
    assert np.allclose(result["boxes"][1], [0.0, 0.0, 0.0, 0.0])


def test_filter_masks_by_area_filters_parallel_tensors():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=bool,
            ),
            "scores": np.array([0.2, 0.9], dtype=np.float32),
            "classes": np.array([1, 2], dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert result["masks"].shape[0] == 1
    assert np.allclose(result["scores"], [0.9])
    assert result["classes"].tolist() == [2]


def test_filter_masks_by_area_handles_empty_masks():
    registry = TensorRegistry(
        {
            "masks": np.zeros((0, 2, 2), dtype=bool),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)

    assert result["masks"].shape == (0, 2, 2)
    assert result["scores"].shape == (0,)
    assert result["classes"].shape == (0,)


def test_empty_instance_postprocess_pipeline_returns_empty_segmentations():
    pipeline = Pipeline([
        TopKIndices2D(
            "class_probs",
            k=100,
            values_as="top_scores",
            row_indices_as="query_indices",
            col_indices_as="class_ids",
        ),
        SelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
        BinarizeTensorByThreshold("selected_masks", threshold=0.5, as_="binary_masks"),
        MeanMaskScores(masks="selected_masks", as_="mean_mask_scores"),
        MultiplyTensors("top_scores", "mean_mask_scores", as_="final_scores"),
        FilterTensorsByMasksArea("final_scores", "class_ids", masks="binary_masks", min_area=1),
        FilterTensorsByScore("binary_masks", "class_ids", score="final_scores", min_score=0.5),
        SortTensorsBy("binary_masks", "class_ids", by="final_scores"),
        MasksToBoxes(masks="binary_masks", as_="boxes"),
        ToSegmentations(scores="final_scores", classes="class_ids", masks="binary_masks"),
    ])
    registry = TensorRegistry(
        {
            "class_probs": np.zeros((0, 3), dtype=np.float32),
            "mask_probs": np.zeros((0, 2, 2), dtype=np.float32),
        }
    )

    result = pipeline(registry)

    assert isinstance(result, Segmentations)
    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []
    assert result.masks == []


def test_create_tensor_mask_writes_boolean_mask_from_predicate():
    registry = TensorRegistry(
        {
            "scores": np.array([0.2, 0.8, 0.5], dtype=np.float32),
        }
    )

    result = CreateTensorMask("scores", predicate=lambda tensor: tensor >= 0.5, as_="keep")(registry)

    assert result["keep"].dtype == np.bool_
    assert result["keep"].tolist() == [False, True, True]


def test_create_tensor_mask_by_threshold_writes_boolean_mask():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[0.4, 0.7], [0.9, 0.1]],
                    [[0.2, 0.3], [0.8, 0.6]],
                ],
                dtype=np.float32,
            )
        }
    )

    result = CreateTensorMaskByThreshold("masks", threshold=0.5, as_="binary_masks")(registry)

    assert result["binary_masks"].dtype == np.bool_
    assert result["binary_masks"].tolist() == [
        [[False, True], [True, False]],
        [[False, False], [True, True]],
    ]


def test_binarize_tensor_by_threshold_is_alias():
    assert BinarizeTensorByThreshold is CreateTensorMaskByThreshold


def test_binarize_tensor_is_alias():
    assert BinarizeTensor is CreateTensorMask


def test_sort_tensors_by_sorts_parallel_tensors():
    registry = TensorRegistry(
        {
            "scores": np.array([0.5, 0.9, 0.1], dtype=np.float32),
            "classes": np.array([5, 9, 1], dtype=np.int64),
        }
    )

    result = SortTensorsBy("classes", by="scores", descending=True)(registry)

    assert np.allclose(result["scores"], [0.9, 0.5, 0.1])
    assert result["classes"].tolist() == [9, 5, 1]


def test_filter_tensors_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "scores": np.array([0.9, 0.5, 0.8], dtype=np.float32),
            "classes": np.array([0, 1, 0], dtype=np.int64),
        }
    )

    result = FilterTensors(
        "scores",
        "classes",
        by="classes",
        predicate=lambda classes: classes == 0,
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert np.allclose(result["selected_scores"], [0.9, 0.8])
    assert result["selected_classes"].tolist() == [0, 0]
    assert np.allclose(result["scores"], [0.9, 0.5, 0.8])


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


def test_filter_tensors_by_masks_area_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "masks": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=bool,
            ),
            "scores": np.array([0.2, 0.9], dtype=np.float32),
            "classes": np.array([1, 2], dtype=np.int64),
        }
    )

    result = FilterTensorsByMasksArea(
        "scores",
        "classes",
        masks="masks",
        min_area=2,
        as_=("selected_masks", "selected_scores", "selected_classes"),
    )(registry)

    assert result["selected_masks"].shape[0] == 1
    assert np.allclose(result["selected_scores"], [0.9])
    assert result["selected_classes"].tolist() == [2]
    assert np.allclose(result["scores"], [0.2, 0.9])


def test_sort_tensors_by_can_write_to_new_keys():
    registry = TensorRegistry(
        {
            "scores": np.array([0.5, 0.9, 0.1], dtype=np.float32),
            "classes": np.array([5, 9, 1], dtype=np.int64),
        }
    )

    result = SortTensorsBy(
        "classes",
        by="scores",
        as_=("sorted_scores", "sorted_classes"),
    )(registry)

    assert np.allclose(result["sorted_scores"], [0.9, 0.5, 0.1])
    assert result["sorted_classes"].tolist() == [9, 5, 1]
    assert np.allclose(result["scores"], [0.5, 0.9, 0.1])


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
# SelectTensors
# ---------------------------------------------------------------------------

def test_select_tensors_synchronises_extra_tensor_with_nms_kept_indices():
    registry = _make_registry(
        boxes=[[10, 10, 50, 50], [12, 12, 48, 48]],
        scores=[0.95, 0.85],
        classes=[0, 0],
    )
    registry["coefficients"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = NMS(kept_as="kept")(registry)
    result = SelectTensors("coefficients", indices="kept")(result)

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

    assert isinstance(result["masks"], np.ndarray)
    assert result["masks"].shape == (1, 2, 2)
    assert result["masks"][0].shape == (2, 2)


def test_project_masks_returns_empty_mask_array_for_empty_input():
    transform = ResizeTransform(
        scale=(2.0, 2.0),
        pad=(0.0, 0.0),
        original_shape=(2, 3),
        resized_shape=(4, 6),
    )
    registry = TensorRegistry()
    registry["boxes"] = np.zeros((0, 4), dtype=np.float32)
    registry["masks"] = np.zeros((0, 4, 4), dtype=np.float32)

    result = ProjectMasks(mask_threshold=0.0)(registry, transform)

    assert isinstance(result["masks"], np.ndarray)
    assert result["masks"].shape == (0, 2, 3)
    assert result["masks"].dtype == np.uint8


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
    assert isinstance(result.masks[0], np.ndarray)


# ---------------------------------------------------------------------------
# AsType
# ---------------------------------------------------------------------------

def test_as_type_can_cast_tuple_of_tensor_payloads():
    tensors = (
        TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        TensorPayload(array=np.array([[3.0, 4.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
    )

    result = AsType("float32")(tensors)

    assert isinstance(result, tuple)
    assert result[0].array.dtype == np.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == np.float32
    assert result[1].dtype == "float32"


def test_as_type_can_cast_list_of_tensor_payloads():
    tensors = [
        TensorPayload(array=np.array([[1.0, 2.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
        TensorPayload(array=np.array([[3.0, 4.0]], dtype=np.float16), layout="UNKNOWN", dtype="float16"),
    ]

    result = AsType("float32")(tensors)

    assert isinstance(result, list)
    assert result[0].array.dtype == np.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == np.float32
    assert result[1].dtype == "float32"


def test_as_type_can_cast_single_tensor_payload():
    payload = TensorPayload(
        array=np.array([[1.0, 2.0]], dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = AsType("float16")(payload)

    assert result.array.dtype == np.float16
    assert result.dtype == "float16"


def test_as_type_can_cast_single_array() -> None:
    array = np.array([[1.0, 2.0]], dtype=np.float16)

    result = AsType("float32")(array)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_as_type_can_cast_named_registry_tensor_in_place() -> None:
    registry = TensorRegistry({"density": np.array([[1.0, 2.0]], dtype=np.float16)})

    result = AsType(src="density", dtype="float32")(registry)

    assert result is registry
    assert result["density"].dtype == np.float32


def test_as_type_can_write_named_registry_tensor_to_new_key() -> None:
    registry = TensorRegistry({"density": np.array([[1.0, 2.0]], dtype=np.float16)})

    result = AsType(src="density", dtype="float32", as_="density_fp32")(registry)

    assert result is registry
    assert result["density"].dtype == np.float16
    assert result["density_fp32"].dtype == np.float32


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


def test_draw_masks_draws_on_source_image():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    source = ImagePayload(array=image, color_space="BGR", layout="HWC")
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    segmentations = Segmentations(
        boxes=[[8.0, 8.0, 24.0, 24.0]],
        scores=[0.9],
        classes=[1],
        masks=[mask],
    )

    result, returned_segmentations = DrawMasks(alpha=0.6)(source, segmentations)

    assert result.array.shape == image.shape
    assert result.color_space == "BGR"
    assert result.layout == "HWC"
    assert np.any(result.array != 0)
    assert returned_segmentations is segmentations


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


# ---------------------------------------------------------------------------
# MapPredictionsToObjects / LogDetections
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SelectTensors / ApplyTensorMask / FilterTensors
# ---------------------------------------------------------------------------

def _registry(**arrays: np.ndarray) -> TensorRegistry:
    r = TensorRegistry()
    for k, v in arrays.items():
        r[k] = v
    return r


def test_select_tensors_applies_index_array():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        kept=np.array([0, 2]),
    )
    result = SelectTensors("scores", indices="kept")(r)
    assert result["scores"].tolist() == [0.9, 0.8]


def test_select_tensors_writes_to_new_key():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        kept=np.array([1]),
    )
    result = SelectTensors("scores", indices="kept", as_="selected_scores")(r)
    assert result["selected_scores"].tolist() == [0.5]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_select_tensors_can_write_multiple_outputs():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([4, 5, 6]),
        kept=np.array([2, 0]),
    )
    result = SelectTensors(
        "scores",
        "classes",
        indices="kept",
        as_=("selected_scores", "selected_classes"),
    )(r)
    assert result["selected_scores"].tolist() == [0.8, 0.9]
    assert result["selected_classes"].tolist() == [6, 4]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_apply_tensor_mask_applies_boolean_mask():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        keep=np.array([True, False, True]),
        classes=np.array([4, 5, 6]),
    )
    result = ApplyTensorMask("scores", "classes", mask="keep")(r)
    assert result["scores"].tolist() == [0.9, 0.8]
    assert result["classes"].tolist() == [4, 6]


def test_apply_tensor_mask_can_write_to_new_keys():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        keep=np.array([True, False, True]),
        classes=np.array([4, 5, 6]),
    )
    result = ApplyTensorMask("scores", "classes", mask="keep", as_=("selected_scores", "selected_classes"))(r)
    assert result["selected_scores"].tolist() == [0.9, 0.8]
    assert result["selected_classes"].tolist() == [4, 6]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_filter_tensors_applies_predicate():
    r = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([0, 1, 0]),
    )
    result = FilterTensors("scores", by="classes", predicate=lambda classes: classes == 0)(r)
    assert result["scores"].tolist() == [0.9, 0.8]


def test_filter_tensors_applies_to_multiple_keys():
    r = _registry(
        boxes=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]),
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([0, 1, 0]),
    )
    result = FilterTensors(
        "boxes",
        "scores",
        "classes",
        by="classes",
        predicate=lambda classes: classes == 0,
    )(r)
    assert result["scores"].tolist() == [0.9, 0.8]
    assert result["classes"].tolist() == [0, 0]
    assert len(result["boxes"]) == 2


def test_filter_tensors_by_score_applies_score_predicate():
    r = _registry(
        boxes=np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]]),
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([0, 1, 0]),
    )

    result = FilterTensorsByScore("boxes", "classes", score="scores", min_score=0.75)(r)

    assert result["scores"].tolist() == [0.9, 0.8]
    assert result["classes"].tolist() == [0, 0]
    assert len(result["boxes"]) == 2


# ---------------------------------------------------------------------------
# FilterPredictions
# ---------------------------------------------------------------------------

def _detections(**kwargs) -> Detections:
    defaults = dict(boxes=[[0,0,1,1],[1,1,2,2],[2,2,3,3]], scores=[0.9,0.5,0.8], classes=[0,1,0])
    defaults.update(kwargs)
    return Detections(**defaults)

def _segmentations() -> Segmentations:
    masks = [np.zeros((4,4), dtype=bool) for _ in range(3)]
    return Segmentations(boxes=[[0,0,1,1],[1,1,2,2],[2,2,3,3]], scores=[0.9,0.5,0.8], classes=[0,1,0], masks=masks)


def test_filter_predictions_generic_predicate():
    d = _detections()
    result = FilterPredictions(predicate=lambda p: [c == 0 for c in p.classes])(d)
    assert result.classes == [0, 0]
    assert result.scores == [0.9, 0.8]


def test_filter_predictions_preserves_subclass_type():
    s = _segmentations()
    result = FilterPredictions(predicate=lambda p: [c == 0 for c in p.classes])(s)
    assert type(result) is Segmentations


def test_filter_predictions_slices_all_fields():
    s = _segmentations()
    result = FilterPredictions(predicate=lambda p: [c == 0 for c in p.classes])(s)
    assert len(result.masks) == 2
    assert len(result.boxes) == 2


def test_filter_predictions_by_class():
    d = _detections()
    result = FilterPredictionsByClass({0})(d)
    assert result.classes == [0, 0]


def test_filter_predictions_by_score():
    d = _detections()
    result = FilterPredictionsByScore(min_score=0.7)(d)
    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_filter_predictions_by_area_min():
    d = Detections(boxes=[[0,0,5,5],[0,0,1,1]], scores=[0.9,0.8], classes=[0,0])
    result = FilterPredictionsByArea(min_area=10)(d)
    assert len(result.boxes) == 1
    assert result.boxes[0] == [0,0,5,5]


def test_filter_predictions_by_area_max():
    d = Detections(boxes=[[0,0,5,5],[0,0,1,1]], scores=[0.9,0.8], classes=[0,0])
    result = FilterPredictionsByArea(max_area=2)(d)
    assert len(result.boxes) == 1
    assert result.boxes[0] == [0,0,1,1]


# ---------------------------------------------------------------------------
# Prediction.filter / Prediction.select — variants
# ---------------------------------------------------------------------------

def test_filter_bool_list():
    d = _detections()
    result = d.filter([True, False, True])
    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_filter_bool_list_all_false():
    d = _detections()
    result = d.filter([False, False, False])
    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_filter_bool_list_all_true():
    d = _detections()
    result = d.filter([True, True, True])
    assert result.scores == d.scores
    assert result.classes == d.classes


def test_filter_numpy_bool_array():
    d = _detections()
    mask = np.array([True, False, True])
    result = d.filter(mask)
    assert result.scores == [0.9, 0.8]


def test_select_index_list():
    d = _detections()
    result = d.select([0, 2])
    assert result.scores == [0.9, 0.8]
    assert result.classes == [0, 0]


def test_select_index_list_single():
    d = _detections()
    result = d.select([1])
    assert result.scores == [0.5]
    assert result.classes == [1]


def test_select_numpy_index_array():
    d = _detections()
    result = d.select(np.array([0, 2]))
    assert result.scores == [0.9, 0.8]


def test_select_numpy_argsort():
    d = _detections()
    # argsort ascending by score: [1, 2, 0] → keep top-2 → indices [2, 0]
    indices = np.argsort(d.scores)[-2:]
    result = d.select(indices)
    assert set(result.scores) == {0.9, 0.8}


def test_filter_empty_mask():
    d = _detections()
    result = d.filter([])
    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []


def test_select_index_out_of_bounds_raises():
    d = _detections()
    with pytest.raises(IndexError):
        d.select([0, 99])


def test_filter_bool_mask_too_long_raises():
    d = _detections()
    with pytest.raises(IndexError):
        d.filter([True, False, True, True])  # 4 elements, only 3 predictions
