from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.tensor import (
    ApplyTensorMask,
    ArgMax,
    BinarizeTensor,
    BinarizeTensorByThreshold,
    CreateTensorMask,
    CreateTensorMaskByThreshold,
    FilterTensors,
    GatherScores,
    MultiplyTensors,
    SelectTensors,
    Sigmoid,
    Slice,
    Softmax,
    SortTensorsBy,
    Squeeze,
    TensorRegistry,
    TopK,
    TopKIndices2D,
    Transpose,
)


def _registry(**arrays: np.ndarray) -> TensorRegistry:
    registry = TensorRegistry()
    for name, array in arrays.items():
        registry[name] = array
    return registry


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


def test_create_tensor_mask_writes_boolean_mask_from_predicate():
    registry = TensorRegistry({"scores": np.array([0.2, 0.8, 0.5], dtype=np.float32)})

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


def test_select_tensors_applies_index_array():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        kept=np.array([0, 2]),
    )

    result = SelectTensors("scores", indices="kept")(registry)

    assert result["scores"].tolist() == [0.9, 0.8]


def test_select_tensors_writes_to_new_key():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        kept=np.array([1]),
    )

    result = SelectTensors("scores", indices="kept", as_="selected_scores")(registry)

    assert result["selected_scores"].tolist() == [0.5]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_select_tensors_can_write_multiple_outputs():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([4, 5, 6]),
        kept=np.array([2, 0]),
    )

    result = SelectTensors(
        "scores",
        "classes",
        indices="kept",
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert result["selected_scores"].tolist() == [0.8, 0.9]
    assert result["selected_classes"].tolist() == [6, 4]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_select_tensors_rejects_boolean_mask():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        keep=np.array([True, False, True]),
    )

    with pytest.raises(TypeError):
        SelectTensors("scores", indices="keep")(registry)


def test_apply_tensor_mask_applies_boolean_mask():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        keep=np.array([True, False, True]),
        classes=np.array([4, 5, 6]),
    )

    result = ApplyTensorMask("scores", "classes", mask="keep")(registry)

    assert result["scores"].tolist() == [0.9, 0.8]
    assert result["classes"].tolist() == [4, 6]


def test_apply_tensor_mask_can_write_to_new_keys():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        keep=np.array([True, False, True]),
        classes=np.array([4, 5, 6]),
    )

    result = ApplyTensorMask("scores", "classes", mask="keep", as_=("selected_scores", "selected_classes"))(registry)

    assert result["selected_scores"].tolist() == [0.9, 0.8]
    assert result["selected_classes"].tolist() == [4, 6]
    assert result["scores"].tolist() == [0.9, 0.5, 0.8]


def test_filter_tensors_applies_predicate():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([0, 1, 0]),
    )

    result = FilterTensors("scores", by="classes", predicate=lambda classes: classes == 0)(registry)

    assert result["scores"].tolist() == [0.9, 0.8]


def test_filter_tensors_applies_to_multiple_keys():
    registry = _registry(
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
    )(registry)

    assert result["scores"].tolist() == [0.9, 0.8]
    assert result["classes"].tolist() == [0, 0]
    assert len(result["boxes"]) == 2


def test_filter_tensors_rejects_integer_index_output():
    registry = _registry(
        scores=np.array([0.9, 0.5, 0.8]),
        classes=np.array([0, 1, 0]),
    )

    with pytest.raises(TypeError):
        FilterTensors(
            "scores",
            by="scores",
            predicate=lambda scores: np.argsort(scores)[-2:],
        )(registry)
