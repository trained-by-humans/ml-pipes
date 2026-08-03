from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml_pipes.torch import (
    TorchArgMax,
    TorchApplyTensorMask,
    TorchAsType,
    TorchBinarizeTensor,
    TorchBinarizeTensorByThreshold,
    TorchCreateTensorMask,
    TorchCreateTensorMaskByThreshold,
    TorchFilterTensors,
    TorchGatherScores,
    TorchMapTensor,
    TorchMultiplyTensors,
    TorchScale,
    TorchSelectTensors,
    TorchSigmoid,
    TorchSqueeze,
    TorchSlice,
    TorchSortTensorsBy,
    TorchSoftmax,
    TorchTopK,
    TorchTopKIndices2D,
    TorchTranspose,
)
from ml_pipes.torch.types import TorchTensorPayload, TorchTensorRegistry


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TorchTensorPayload:
    return TorchTensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def test_torch_as_type_supports_payload_registry_and_sequence_forms():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")
    cast_payload = TorchAsType("float16")(payload)
    assert cast_payload.dtype == "float16"

    registry = TorchTensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})
    TorchAsType("float16", src="scores")(registry)
    assert registry["scores"].dtype == torch.float16

    sequence = TorchAsType("float16")([torch.ones((1,), dtype=torch.float32)])
    assert sequence[0].dtype == torch.float16


def test_torch_as_type_can_cast_tuple_of_tensor_payloads():
    tensors = (
        _torch_payload(torch.tensor([[1.0, 2.0]], dtype=torch.float16), layout="UNKNOWN"),
        _torch_payload(torch.tensor([[3.0, 4.0]], dtype=torch.float16), layout="UNKNOWN"),
    )

    result = TorchAsType("float32")(tensors)

    assert isinstance(result, tuple)
    assert result[0].array.dtype == torch.float32
    assert result[0].dtype == "float32"
    assert result[1].array.dtype == torch.float32
    assert result[1].dtype == "float32"


def test_torch_as_type_can_cast_single_tensor():
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    result = TorchAsType("float32")(tensor)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_torch_as_type_can_write_named_registry_tensor_to_new_key():
    registry = TorchTensorRegistry({"density": torch.tensor([[1.0, 2.0]], dtype=torch.float16)})

    result = TorchAsType(src="density", dtype="float32", as_="density_fp32")(registry)

    assert result is registry
    assert result["density"].dtype == torch.float16
    assert result["density_fp32"].dtype == torch.float32


def test_torch_as_type_without_src_rejects_registry_input():
    registry = TorchTensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})

    with pytest.raises(TypeError):
        TorchAsType("float16")(registry)


def test_torch_as_type_with_src_rejects_payload_input():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")

    with pytest.raises(TypeError):
        TorchAsType("float16", src="scores")(payload)


def test_torch_as_type_rejects_as_without_src():
    with pytest.raises(ValueError):
        TorchAsType("float16", as_="scores_fp16")


def test_torch_argmax_and_multiply_tensors_work_on_registry():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
            "left": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            "right": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        }
    )

    TorchArgMax("scores", as_="classes")(registry)
    TorchMultiplyTensors("left", "right", as_="product")(registry)

    assert registry["classes"].tolist() == [1, 0]
    assert torch.allclose(registry["product"], torch.tensor([[10.0, 20.0], [20.0, 40.0]]))


def test_torch_scale_multiplies_by_broadcastable_factors():
    registry = TorchTensorRegistry({"preds": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)})

    result = TorchScale("preds", by=(10.0, 100.0), as_="scaled_preds")(registry)

    assert torch.allclose(result["scaled_preds"], torch.tensor([[10.0, 200.0], [30.0, 400.0]]))
    assert torch.allclose(result["preds"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_torch_squeeze_removes_size_one_batch_dim():
    registry = TorchTensorRegistry({"preds": torch.zeros((1, 5, 10), dtype=torch.float32)})

    result = TorchSqueeze("preds")(registry)

    assert tuple(result["preds"].shape) == (5, 10)


def test_torch_transpose_swaps_axes():
    registry = TorchTensorRegistry({"preds": torch.zeros((5, 10), dtype=torch.float32)})

    result = TorchTranspose("preds")(registry)

    assert tuple(result["preds"].shape) == (10, 5)


def test_torch_squeeze_supports_single_and_multiple_axes():
    registry = TorchTensorRegistry({"preds": torch.zeros((1, 2, 1, 3), dtype=torch.float32)})

    TorchSqueeze("preds", axis=(0, 2))(registry)

    assert tuple(registry["preds"].shape) == (2, 3)


def test_torch_squeeze_rejects_non_unit_axis():
    registry = TorchTensorRegistry({"preds": torch.zeros((1, 2, 1, 3), dtype=torch.float32)})

    with pytest.raises(ValueError, match="cannot squeeze axis"):
        TorchSqueeze("preds", axis=1)(registry)


def test_torch_argmax_axis_zero_handles_empty_leading_dimension():
    registry = TorchTensorRegistry({"scores": torch.zeros((0, 2, 3), dtype=torch.float32)})

    TorchArgMax("scores", axis=0, as_="classes")(registry)

    assert registry["classes"].dtype == torch.int64
    assert tuple(registry["classes"].shape) == (2, 3)
    assert torch.equal(registry["classes"], torch.zeros((2, 3), dtype=torch.int64))


def test_torch_gather_scores_picks_class_score():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
            "classes": torch.tensor([1, 0], dtype=torch.int64),
        }
    )

    result = TorchGatherScores("scores", "classes")(registry)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))


def test_torch_softmax_slice_sigmoid_argmax_and_gather_scores_work_on_registry():
    registry = TorchTensorRegistry(
        {
            "class_logits": torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.5, -1.0]], dtype=torch.float32),
            "mask_logits": torch.tensor(
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                ],
                dtype=torch.float32,
            ),
        }
    )

    TorchSoftmax("class_logits", as_="class_probs")(registry)
    TorchSlice("class_probs", slice(None, -1))(registry)
    TorchSigmoid("mask_logits", as_="mask_probs")(registry)
    TorchArgMax("class_probs", as_="query_classes")(registry)
    TorchGatherScores("class_probs", "query_classes", as_="query_scores")(registry)

    assert registry["class_probs"].shape == (2, 2)
    assert registry["mask_probs"].shape == (2, 2, 2)
    assert registry["query_classes"].dtype == torch.int64
    assert registry["query_scores"].shape == (2,)


def test_torch_slice_extracts_column_range():
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    registry = TorchTensorRegistry({"preds": data})

    result = TorchSlice("preds", at=slice(None, 2), as_="boxes")(registry)

    assert tuple(result["boxes"].shape) == (3, 2)
    assert torch.equal(result["boxes"], data[:, :2])


def test_torch_softmax_sums_to_one_per_row():
    registry = TorchTensorRegistry({"logits": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)})

    result = TorchSoftmax("logits")(registry)

    assert torch.allclose(result["logits"].sum(dim=-1), torch.tensor([1.0]))


def test_torch_softmax_handles_empty_reduction_axis():
    registry = TorchTensorRegistry({"logits": torch.zeros((0, 2, 3), dtype=torch.float32)})

    result = TorchSoftmax("logits", axis=0)(registry)

    assert tuple(result["logits"].shape) == (0, 2, 3)
    assert result["logits"].dtype == torch.float32


def test_torch_sigmoid_maps_zero_to_half():
    registry = TorchTensorRegistry({"x": torch.tensor([[0.0]], dtype=torch.float32)})

    result = TorchSigmoid("x")(registry)

    assert torch.allclose(result["x"], torch.tensor([[0.5]], dtype=torch.float32))


def test_torch_sigmoid_is_stable_for_large_magnitude_values():
    registry = TorchTensorRegistry({"x": torch.tensor([[-1000.0, 1000.0]], dtype=torch.float32)})

    result = TorchSigmoid("x")(registry)

    assert torch.isfinite(result["x"]).all()
    assert result["x"].dtype == torch.float32
    assert torch.allclose(result["x"], torch.tensor([[0.0, 1.0]], dtype=torch.float32))


def test_torch_topk_and_topk_indices_2d_return_expected_values_and_indices():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.1, 0.7, 0.2, 0.8], dtype=torch.float32),
            "class_probs": torch.tensor(
                [
                    [0.1, 0.7, 0.2],
                    [0.8, 0.3, 0.6],
                ],
                dtype=torch.float32,
            ),
        }
    )

    TorchTopK("scores", k=3, values_as="top_scores", indices_as="top_indices")(registry)
    TorchTopKIndices2D(
        "class_probs",
        k=3,
        values_as="pair_scores",
        row_indices_as="query_indices",
        col_indices_as="class_ids",
    )(registry)

    assert torch.allclose(registry["top_scores"], torch.tensor([0.8, 0.7, 0.2]))
    assert registry["top_indices"].tolist() == [3, 1, 2]
    assert torch.allclose(registry["pair_scores"], torch.tensor([0.8, 0.7, 0.6]))
    assert registry["query_indices"].tolist() == [1, 0, 1]
    assert registry["class_ids"].tolist() == [0, 1, 2]


def test_torch_sort_tensors_by_sorts_parallel_tensors():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.5, 0.9, 0.1], dtype=torch.float32),
            "classes": torch.tensor([5, 9, 1], dtype=torch.int64),
        }
    )

    result = TorchSortTensorsBy("classes", by="scores", descending=True)(registry)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.5, 0.1]))
    assert result["classes"].tolist() == [9, 5, 1]


def test_torch_filter_tensors_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    result = TorchFilterTensors(
        "scores",
        "classes",
        by="classes",
        predicate=lambda classes: classes == 0,
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert torch.allclose(result["selected_scores"], torch.tensor([0.9, 0.8]))
    assert result["selected_classes"].tolist() == [0, 0]
    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_sort_tensors_by_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.5, 0.9, 0.1], dtype=torch.float32),
            "classes": torch.tensor([5, 9, 1], dtype=torch.int64),
        }
    )

    TorchSortTensorsBy("classes", by="scores", as_=("sorted_scores", "sorted_classes"))(registry)

    assert torch.allclose(registry["sorted_scores"], torch.tensor([0.9, 0.5, 0.1]))
    assert registry["sorted_classes"].tolist() == [9, 5, 1]
    assert torch.allclose(registry["scores"], torch.tensor([0.5, 0.9, 0.1]))


def test_torch_apply_tensor_mask_applies_boolean_mask():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "keep": torch.tensor([True, False, True], dtype=torch.bool),
            "classes": torch.tensor([4, 5, 6], dtype=torch.int64),
        }
    )

    result = TorchApplyTensorMask("scores", "classes", mask="keep")(registry)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))
    assert result["classes"].tolist() == [4, 6]


def test_torch_create_tensor_mask_writes_boolean_mask_from_predicate():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
        }
    )

    result = TorchCreateTensorMask("scores", predicate=lambda tensor: tensor >= 0.5, as_="keep")(registry)

    assert result["keep"].dtype == torch.bool
    assert result["keep"].tolist() == [False, True, True]


def test_torch_create_tensor_mask_places_numpy_predicate_output_on_source_device():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
        }
    )

    result = TorchCreateTensorMask(
        "scores",
        predicate=lambda tensor: tensor.detach().cpu().numpy() >= 0.5,
        as_="keep",
    )(registry)

    assert result["keep"].dtype == torch.bool
    assert result["keep"].device == result["scores"].device
    assert result["keep"].tolist() == [False, True, True]


def test_torch_create_tensor_mask_by_threshold_writes_boolean_mask():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[0.4, 0.7], [0.9, 0.1]],
                    [[0.2, 0.3], [0.8, 0.6]],
                ],
                dtype=torch.float32,
            )
        }
    )

    result = TorchCreateTensorMaskByThreshold("masks", threshold=0.5, as_="binary_masks")(registry)

    assert result["binary_masks"].dtype == torch.bool
    assert result["binary_masks"].tolist() == [
        [[False, True], [True, False]],
        [[False, False], [True, True]],
    ]


def test_torch_binarize_tensor_by_threshold_is_alias():
    assert TorchBinarizeTensorByThreshold is TorchCreateTensorMaskByThreshold


def test_torch_binarize_tensor_is_alias():
    assert TorchBinarizeTensor is TorchCreateTensorMask


def test_torch_select_tensors_and_apply_tensor_mask_work_on_registry():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([4, 5, 6], dtype=torch.int64),
            "indices": torch.tensor([2, 0], dtype=torch.int64),
            "keep": torch.tensor([True, False], dtype=torch.bool),
        }
    )

    TorchSelectTensors("scores", "classes", indices="indices")(registry)
    TorchApplyTensorMask("scores", "classes", mask="keep")(registry)

    assert torch.allclose(registry["scores"], torch.tensor([0.8]))
    assert registry["classes"].tolist() == [6]


def test_torch_apply_tensor_mask_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([4, 5, 6], dtype=torch.int64),
            "keep": torch.tensor([True, False, True], dtype=torch.bool),
        }
    )

    TorchApplyTensorMask(
        "scores",
        "classes",
        mask="keep",
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9, 0.8]))
    assert registry["selected_classes"].tolist() == [4, 6]
    assert torch.allclose(registry["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_select_tensors_writes_to_new_key():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "indices": torch.tensor([1], dtype=torch.int64),
        }
    )

    TorchSelectTensors("scores", indices="indices", as_="selected_scores")(registry)

    assert torch.allclose(registry["selected_scores"], torch.tensor([0.5]))
    assert torch.allclose(registry["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_select_tensors_rejects_boolean_mask():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "keep": torch.tensor([True, False, True], dtype=torch.bool),
        }
    )

    with pytest.raises(TypeError):
        TorchSelectTensors("scores", indices="keep")(registry)


def test_torch_select_tensors_can_write_multiple_outputs():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([4, 5, 6], dtype=torch.int64),
            "indices": torch.tensor([2, 0], dtype=torch.int64),
        }
    )

    TorchSelectTensors(
        "scores",
        "classes",
        indices="indices",
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert torch.allclose(registry["selected_scores"], torch.tensor([0.8, 0.9]))
    assert registry["selected_classes"].tolist() == [6, 4]
    assert torch.allclose(registry["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_filter_tensors_applies_predicate():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    result = TorchFilterTensors("scores", by="classes", predicate=lambda classes: classes == 0)(registry)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))


def test_torch_filter_tensors_applies_to_multiple_keys():
    registry = TorchTensorRegistry(
        {
            "boxes": torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]], dtype=torch.float32),
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    result = TorchFilterTensors(
        "boxes",
        "scores",
        "classes",
        by="classes",
        predicate=lambda classes: classes == 0,
    )(registry)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))
    assert result["classes"].tolist() == [0, 0]
    assert len(result["boxes"]) == 2


def test_torch_filter_tensors_rejects_integer_index_output():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    with pytest.raises(TypeError):
        TorchFilterTensors(
            "scores",
            by="scores",
            predicate=lambda scores: torch.argsort(scores)[-2:],
        )(registry)


def test_torch_map_tensor_applies_fn():
    registry = TorchTensorRegistry({"mask_probs": torch.tensor([[0.1, 0.9]], dtype=torch.float32)})

    result = TorchMapTensor("mask_probs", fn=lambda tensor: tensor + 1.0, as_="mapped_mask_probs")(registry)

    assert torch.allclose(result["mapped_mask_probs"], torch.tensor([[1.1, 1.9]], dtype=torch.float32))
    assert torch.allclose(result["mask_probs"], torch.tensor([[0.1, 0.9]], dtype=torch.float32))
