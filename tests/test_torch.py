from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes import Batch, Gather, Normalize, Pipeline, PipelineValidationError, Scatter, ToSegmentations, TracingConfig, UnBatch
from ml_pipes.inspection import ImageBlock, PipelineInspector, TextBlock
from ml_pipes.tracing import TraceCollector
from ml_pipes.types import TensorPayload
from ml_pipes.torch import (
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
    TorchArgMax,
    TorchApplyTensorMask,
    TorchAsType,
    TorchBinarizeTensor,
    TorchBinarizeTensorByThreshold,
    TorchCollate,
    TorchCreateTensorMask,
    TorchCreateTensorMaskByThreshold,
    TorchDistribute,
    TorchExtract,
    TorchFilterTensorsByClasses,
    TorchFilterTensorsByMasksArea,
    TorchFilterTensorsByScore,
    TorchGatherRows,
    TorchGatherScores,
    TorchInfer,
    TorchMasksToBoxes,
    TorchMeanMaskScores,
    TorchMultiplyTensors,
    TorchNMS,
    TorchResizeMasks,
    TorchSelectTensors,
    TorchSigmoid,
    TorchSlice,
    TorchSortTensorsBy,
    TorchSoftmax,
    TorchSynchronizeTensors,
    TorchTopK,
    TorchTopKIndices2D,
    TorchWeightMasksByScores,
)
from ml_pipes.torch.ops import _numpy_conversion_can_alias_torch_source, _torch_conversion_can_alias_numpy_source
from ml_pipes.torch.types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry


class _CaptureCollector(TraceCollector):
    def __init__(self) -> None:
        self.traces = []

    def on_trace(self, trace) -> None:
        self.traces.append(trace)


class _TorchIdentity:
    def __call__(self, value: TorchTensorPayload) -> TorchTensorPayload:
        return value


class _TorchIncrementRegistry:
    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry["scores"] = registry["scores"] + 1
        return registry


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TorchTensorPayload:
    return TorchTensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def test_numpy_torch_numpy_pipeline_composes():
    pipeline = Pipeline([
        ToTorch(),
        TorchAsType("float16"),
        ToNumpy(),
    ])
    payload = TensorPayload(
        array=np.ones((1, 3, 4, 4), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

    assert isinstance(result, TensorPayload)
    assert result.layout == "NCHW"
    assert result.dtype == "float16"
    assert result.array.dtype == np.float16


def test_validate_mixed_domains_fail_without_explicit_conversion():
    pipeline = Pipeline([
        Normalize(),
        TorchInfer(torch.nn.Identity().eval()),
    ])

    with pytest.raises(PipelineValidationError, match="TorchInfer"):
        pipeline.validate(inference=True)


def test_torch_infer_extract_and_registry_conversion_round_trip():
    class _Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x + 1, x.sum(dim=1)

    pipeline = Pipeline([
        ToTorch(),
        TorchInfer(_Module().eval(), output_names=("boxes", "scores"), output_layouts=("NCHW", "NHW")),
        TorchExtract("scores", as_="scores"),
        ToNumpyRegistry(),
    ])
    payload = TensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

    assert np.array_equal(result["scores"], np.full((1, 2, 2), 3.0, dtype=np.float32))


def test_torch_infer_accepts_sequence_outputs():
    def _infer(x: torch.Tensor) -> list[torch.Tensor]:
        return [x + 1, x.sum(dim=1)]

    payload = _torch_payload(torch.ones((1, 3, 2, 2), dtype=torch.float32))

    result = TorchInfer(
        _infer,
        output_names=("boxes", "scores"),
        output_layouts=("NCHW", "NHW"),
    )(payload)

    assert result.names == ("boxes", "scores")
    assert len(result.tensors) == 2
    assert tuple(result.tensors[0].array.shape) == (1, 3, 2, 2)
    assert tuple(result.tensors[1].array.shape) == (1, 2, 2)


def test_to_torch_copy_false_shares_cpu_numpy_storage():
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(copy=False)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [9.0, 2.0]


def test_to_torch_copy_true_isolates_cpu_numpy_storage():
    payload = TensorPayload(
        array=np.array([1.0, 2.0], dtype=np.float32),
        layout="N",
        dtype="float32",
    )

    result = ToTorch(copy=True)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [1.0, 2.0]


def test_to_numpy_copy_false_shares_cpu_torch_storage():
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32), layout="N")

    result = ToNumpy(copy=False)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [9.0, 2.0]


def test_to_numpy_copy_true_isolates_cpu_torch_storage():
    payload = _torch_payload(torch.tensor([1.0, 2.0], dtype=torch.float32), layout="N")

    result = ToNumpy(copy=True)(payload)
    payload.array[0] = 9.0

    assert result.array.tolist() == [1.0, 2.0]


def test_to_torch_registry_copy_true_isolates_cpu_numpy_storage():
    from ml_pipes.types import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})

    result = ToTorchRegistry(copy=True)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [1.0, 2.0]


def test_to_numpy_registry_copy_false_shares_cpu_torch_storage():
    registry = TorchTensorRegistry({"scores": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    result = ToNumpyRegistry(copy=False)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [9.0, 2.0]


def test_torch_infer_defaults_output_names_and_layouts():
    op = TorchInfer(torch.nn.Identity().eval())
    outputs = op(_torch_payload(torch.ones((1, 3, 4, 4))))

    assert outputs.names == ("output_0",)
    assert outputs.tensors[0].layout == "UNKNOWN"
    assert outputs.tensors[0].dtype == "float32"
    assert outputs.tensors[0].device == "cpu"


def test_torch_extract_raises_for_missing_name():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((1, 3))),),
        names=("present",),
    )

    with pytest.raises(KeyError, match="missing"):
        TorchExtract("missing")(outputs)


def test_torch_collate_matches_numpy_shape_semantics():
    tensors = [
        _torch_payload(torch.zeros((1, 3, 8, 8))),
        _torch_payload(torch.zeros((1, 3, 8, 8))),
    ]

    result = TorchCollate()(tensors)

    assert result.array.shape == (2, 3, 8, 8)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"


def test_torch_distribute_clones_per_sample_outputs():
    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((2, 4))),),
        names=("preds",),
    )

    result = TorchDistribute()(outputs)
    result[0].tensors[0].array[:] = 99

    assert torch.all(result[1].tensors[0].array == 1)
    assert result[0].tensors[0].array.data_ptr() != result[1].tensors[0].array.data_ptr()


def test_torch_as_type_supports_payload_registry_and_sequence_forms():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")
    cast_payload = TorchAsType("float16")(payload)
    assert cast_payload.dtype == "float16"

    registry = TorchTensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})
    TorchAsType("float16", src="scores")(registry)
    assert registry["scores"].dtype == torch.float16

    sequence = TorchAsType("float16")([torch.ones((1,), dtype=torch.float32)])
    assert sequence[0].dtype == torch.float16


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


def test_torch_argmax_axis_zero_handles_empty_leading_dimension():
    registry = TorchTensorRegistry({"scores": torch.zeros((0, 2, 3), dtype=torch.float32)})

    TorchArgMax("scores", axis=0, as_="classes")(registry)

    assert registry["classes"].dtype == torch.int64
    assert tuple(registry["classes"].shape) == (2, 3)
    assert torch.equal(registry["classes"], torch.zeros((2, 3), dtype=torch.int64))


def test_empty_torch_instance_postprocess_pipeline_returns_empty_segmentations():
    pipeline = Pipeline([
        TorchTopKIndices2D(
            "class_probs",
            k=100,
            values_as="top_scores",
            row_indices_as="query_indices",
            col_indices_as="class_ids",
        ),
        TorchSelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
        TorchBinarizeTensorByThreshold("selected_masks", threshold=0.5, as_="binary_masks"),
        TorchMeanMaskScores(masks="selected_masks", as_="mean_mask_scores"),
        TorchMultiplyTensors("top_scores", "mean_mask_scores", as_="final_scores"),
        TorchFilterTensorsByMasksArea("final_scores", "class_ids", masks="binary_masks", min_area=1),
        TorchFilterTensorsByScore("binary_masks", "class_ids", score="final_scores", min_score=0.5),
        TorchSortTensorsBy("binary_masks", "class_ids", by="final_scores"),
        TorchMasksToBoxes(masks="binary_masks", as_="boxes"),
        ToNumpyRegistry(),
        ToSegmentations(scores="final_scores", classes="class_ids", masks="binary_masks"),
    ])
    registry = TorchTensorRegistry(
        {
            "class_probs": torch.zeros((0, 3), dtype=torch.float32),
            "mask_probs": torch.zeros((0, 2, 2), dtype=torch.float32),
        }
    )

    result = pipeline(registry)

    assert result.boxes == []
    assert result.scores == []
    assert result.classes == []
    assert result.masks == []


def test_torch_softmax_slice_gather_and_threshold_tensors_work_on_registry():
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
    TorchFilterTensorsByScore("query_classes", "mask_probs", score="query_scores", min_score=0.2)(registry)

    assert registry["class_probs"].shape == (2, 2)
    assert registry["mask_probs"].shape == (2, 2, 2)
    assert registry["query_classes"].dtype == torch.int64
    assert registry["query_scores"].shape == (2,)


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


def test_torch_weight_masks_by_scores_broadcasts_scores_over_masks():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.5, 2.0], dtype=torch.float32),
            "masks": torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ],
                dtype=torch.float32,
            ),
        }
    )

    result = TorchWeightMasksByScores(as_="weighted_masks")(registry)

    assert torch.allclose(
        result["weighted_masks"],
        torch.tensor(
            [
                [[0.5, 1.0], [1.5, 2.0]],
                [[10.0, 12.0], [14.0, 16.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_torch_resize_masks_to_image_resizes_mask_stack():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
    )

    TorchResizeMasks(masks="masks", as_="resized_masks")(registry, (4, 6))

    assert registry["resized_masks"].shape == (1, 4, 6)
    assert registry["resized_masks"].dtype == torch.float32


def test_torch_mean_mask_scores_computes_mean_over_binary_support():
    registry = TorchTensorRegistry(
        {
            "selected_masks": torch.tensor(
                [
                    [[0.0, 1.0], [0.5, 0.0]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ],
                dtype=torch.float32,
            ),
            "binary_masks": torch.tensor(
                [
                    [[False, True], [True, False]],
                    [[True, False], [False, True]],
                ],
                dtype=torch.bool,
            ),
        }
    )

    TorchMeanMaskScores(masks="selected_masks", as_="mean_mask_scores")(registry)

    assert torch.allclose(registry["mean_mask_scores"], torch.tensor([0.75, 0.5]))


def test_torch_masks_to_boxes_converts_masks_to_xyxy():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[False, True, True], [False, True, False]],
                    [[False, False, False], [False, False, False]],
                ],
                dtype=torch.bool,
            )
        }
    )

    TorchMasksToBoxes(as_="boxes")(registry)

    assert torch.allclose(registry["boxes"][0], torch.tensor([1.0, 0.0, 3.0, 2.0]))
    assert torch.allclose(registry["boxes"][1], torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_torch_filter_masks_by_area_and_sort_tensors_by_work_on_registry():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=torch.bool,
            ),
            "scores": torch.tensor([0.5, 0.9], dtype=torch.float32),
            "classes": torch.tensor([5, 9], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByMasksArea("scores", "classes", masks="masks", min_area=2)(registry)
    TorchSortTensorsBy("classes", by="scores", descending=True)(registry)

    assert registry["masks"].shape[0] == 1
    assert torch.allclose(registry["scores"], torch.tensor([0.9]))
    assert registry["classes"].tolist() == [9]


def test_torch_filter_tensors_by_score_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 0], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByScore(
        "classes",
        score="scores",
        min_score=0.75,
        as_=("selected_scores", "selected_classes"),
    )(registry)

    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9, 0.8]))
    assert registry["selected_classes"].tolist() == [0, 0]
    assert torch.allclose(registry["scores"], torch.tensor([0.9, 0.5, 0.8]))


def test_torch_filter_tensors_by_masks_area_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "masks": torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[1, 1], [1, 0]],
                ],
                dtype=torch.bool,
            ),
            "scores": torch.tensor([0.2, 0.9], dtype=torch.float32),
            "classes": torch.tensor([1, 2], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByMasksArea(
        "scores",
        "classes",
        masks="masks",
        min_area=2,
        as_=("selected_masks", "selected_scores", "selected_classes"),
    )(registry)

    assert registry["selected_masks"].shape[0] == 1
    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9]))
    assert registry["selected_classes"].tolist() == [2]
    assert torch.allclose(registry["scores"], torch.tensor([0.2, 0.9]))


def test_torch_filter_tensors_by_classes_can_write_to_new_keys():
    registry = TorchTensorRegistry(
        {
            "scores": torch.tensor([0.9, 0.5, 0.8], dtype=torch.float32),
            "classes": torch.tensor([0, 1, 2], dtype=torch.int64),
        }
    )

    TorchFilterTensorsByClasses(
        "scores",
        classes="classes",
        keep_classes=[0, 2],
        as_=("selected_classes", "selected_scores"),
    )(registry)

    assert registry["selected_classes"].tolist() == [0, 2]
    assert torch.allclose(registry["selected_scores"], torch.tensor([0.9, 0.8]))
    assert registry["classes"].tolist() == [0, 1, 2]


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


def test_to_device_updates_payload_and_registry_devices():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))
    moved_payload = ToDevice("cpu")(payload)
    assert moved_payload.device == "cpu"

    registry = TorchTensorRegistry({"scores": torch.ones((2,), dtype=torch.float32)})
    moved_registry = ToDevice("cpu")(registry)
    assert moved_registry["scores"].device.type == "cpu"


def test_to_device_supports_tensor_sequences_and_runtime_outputs():
    tensor = torch.ones((2,), dtype=torch.float32)
    moved_tensor = ToDevice("cpu")(tensor)
    assert moved_tensor.device.type == "cpu"

    payloads = [_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC")]
    moved_payloads = ToDevice("cpu")(payloads)
    assert moved_payloads[0].device == "cpu"

    outputs = TorchRuntimeOutputs(
        tensors=(_torch_payload(torch.ones((1, 2), dtype=torch.float32), layout="NC"),),
        names=("scores",),
    )
    moved_outputs = ToDevice("cpu")(outputs)
    assert moved_outputs.tensors[0].device == "cpu"


def test_torch_synchronize_tensors_passthrough_on_payload():
    payload = _torch_payload(torch.ones((1, 2), dtype=torch.float32))

    result = TorchSynchronizeTensors()(payload)

    assert result is payload


def test_torch_synchronize_tensors_collects_devices_from_runtime_outputs(monkeypatch):
    outputs = TorchRuntimeOutputs(
        tensors=(
            _torch_payload(torch.ones((1, 2), dtype=torch.float32)),
            _torch_payload(torch.ones((1, 3), dtype=torch.float32)),
        ),
        names=("a", "b"),
    )
    seen: list[str] = []

    monkeypatch.setattr(
        "ml_pipes.torch.ops._synchronize_torch_device",
        lambda device: seen.append(str(device)),
    )

    result = TorchSynchronizeTensors()(outputs)

    assert result is outputs
    assert seen == ["cpu"]


def test_torch_synchronize_tensors_rejects_non_torch_values():
    with pytest.raises(TypeError, match="TorchSynchronizeTensors"):
        TorchSynchronizeTensors()(123)


def test_torch_nms_filters_and_stores_indices():
    pytest.importorskip("torchvision")
    registry = TorchTensorRegistry(
        {
            "boxes": torch.tensor(
                [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32),
            "classes": torch.tensor([0, 0, 1], dtype=torch.int64),
        }
    )

    result = TorchNMS(kept_as="kept", iou_threshold=0.5)(registry)

    assert result["boxes"].shape[0] == 2
    assert result["kept"].dtype == torch.int64
    assert result["kept"].tolist() == [2, 0]


def test_torch_batch_region_validates_and_runs():
    pipeline = Pipeline([
        Batch(size=2, timeout=0.01),
        TorchCollate(),
        TorchInfer(torch.nn.Identity().eval()),
        TorchDistribute(),
        UnBatch(),
    ])
    contract = pipeline.validate(inference=True)

    sample = _torch_payload(torch.ones((1, 3, 2, 2)))
    result = pipeline(sample)

    assert contract.input_type is TorchTensorPayload
    assert isinstance(result, TorchRuntimeOutputs)
    assert result.tensors[0].array.shape == (1, 3, 2, 2)


def test_torch_scatter_region_validates_and_runs():
    pipeline = Pipeline([
        Scatter(max_concurrency=2),
        _TorchIdentity(),
        Gather(),
    ])
    contract = pipeline.validate(inference=True)
    payloads = [_torch_payload(torch.ones((1, 2))) for _ in range(3)]

    result = pipeline(payloads)

    assert contract.input_type == list[TorchTensorPayload]
    assert len(result) == 3
    assert all(isinstance(item, TorchTensorPayload) for item in result)


def test_torch_registry_conversion_handoff_back_to_numpy():
    pipeline = Pipeline([
        ToTorchRegistry(),
        _TorchIncrementRegistry(),
        ToNumpyRegistry(),
    ])
    from ml_pipes.types import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})
    result = pipeline(registry)

    assert np.array_equal(result["scores"], np.array([2.0, 3.0], dtype=np.float32))


def test_to_torch_registry_copy_false_shares_cpu_numpy_storage():
    from ml_pipes.types import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})

    result = ToTorchRegistry(copy=False)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [9.0, 2.0]


def test_to_numpy_registry_copy_true_isolates_cpu_torch_storage():
    registry = TorchTensorRegistry({"scores": torch.tensor([1.0, 2.0], dtype=torch.float32)})

    result = ToNumpyRegistry(copy=True)(registry)
    registry["scores"][0] = 9.0

    assert result["scores"].tolist() == [1.0, 2.0]


def test_pipeline_inspector_formats_torch_tensor_registry_like_tensor_registry():
    registry = TorchTensorRegistry(
        {
            "class_queries_logits": torch.zeros((100, 81), dtype=torch.float32),
            "masks_queries_logits": torch.zeros((100, 96, 96), dtype=torch.float32),
        }
    )

    blocks = PipelineInspector()._output_to_blocks(registry)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "TorchTensorRegistry"
    assert blocks[0].rows == [
        ("class_queries_logits", "(100, 81)@cpu"),
        ("masks_queries_logits", "(100, 96, 96)@cpu"),
    ]


def test_pipeline_inspector_formats_primitive_tuple_as_single_block():
    blocks = PipelineInspector()._output_to_blocks((480, 640))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "tuple"
    assert blocks[0].rows == [("", "(480, 640)")]


def test_pipeline_inspector_formats_rgb_ndarray_as_image():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    blocks = PipelineInspector()._output_to_blocks(image)

    assert len(blocks) == 2
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].title == "ndarray  6×4  RGB"
    assert np.array_equal(blocks[0].array, image)
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].title == "ndarray"
    assert blocks[1].rows == [("shape", "(4, 6, 3)"), ("dtype", "uint8")]


def test_pipeline_inspector_formats_non_image_ndarray_as_text():
    blocks = PipelineInspector()._output_to_blocks(np.zeros((2, 3), dtype=np.float32))

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ndarray"
    assert blocks[0].rows == [("shape", "(2, 3)"), ("dtype", "float32")]


def test_pipeline_inspector_formats_list_of_dicts():
    blocks = PipelineInspector()._output_to_blocks(
        [
            {"class_id": 1, "score": 0.9},
            {"class_id": 2, "score": 0.8},
        ]
    )

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "list  ×2"
    assert blocks[0].rows == [
        ("[0]", "dict  class_id 1  |  score 0.9"),
        ("[1]", "dict  class_id 2  |  score 0.8"),
    ]


def test_torch_tracing_records_device_shapes_and_operator_config():
    collector = _CaptureCollector()
    pipeline = Pipeline(
        [
            ToTorch(device="cpu"),
            ToDevice("cpu"),
            TorchInfer(torch.nn.Identity().eval(), serialize=True),
            TorchExtract("output_0", as_="scores"),
        ],
        tracing=TracingConfig(collector=collector, capture_shapes=True, capture_config=True),
    )
    payload = TensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    pipeline(payload)

    spans = collector.traces[0].spans
    assert spans[0].output_shape == "TorchTensorPayload (1, 3, 2, 2) @ cpu"
    assert spans[1].operator_config["device"] == "cpu"
    assert spans[2].output_shape == "TorchRuntimeOutputs {output_0: (1, 3, 2, 2) @ cpu}"
    assert spans[3].output_shape == "TorchTensorRegistry {scores: (1, 3, 2, 2) @ cpu}"


def test_torch_validation_accepts_to_device_and_torch_as_type_boundaries():
    contract = Pipeline([
        ToTorch(),
        ToDevice("cpu"),
        TorchSynchronizeTensors(),
        TorchAsType("float16"),
        _TorchIdentity(),
    ]).validate(inference=True)

    assert contract.input_type is TensorPayload
    assert contract.output_type is TorchTensorPayload


def test_torch_validation_rejects_registry_op_after_to_device_payload():
    with pytest.raises(PipelineValidationError):
        Pipeline([ToDevice("cpu"), TorchArgMax("scores")]).validate(
            pipeline_input_type=TorchTensorPayload
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_smoke_for_to_torch_and_to_device():
    payload = TensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    to_cuda = ToTorch(device="cuda:0")
    result = ToDevice("cuda:0")(to_cuda(payload))

    assert result.device == "cuda:0"
    assert result.array.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_numpy_copy_true_from_cuda_does_not_need_alias_breaking_copy():
    payload = _torch_payload(torch.ones((2,), dtype=torch.float32, device="cuda:0"), layout="N")

    assert not _numpy_conversion_can_alias_torch_source(
        source_device_type=payload.array.device.type,
        source_dtype=np.dtype("float32"),
        target_dtype=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_to_torch_copy_true_to_cuda_does_not_need_alias_breaking_copy():
    array = np.ones((2,), dtype=np.float32)
    source_dtype = torch.as_tensor(array).dtype

    assert not _torch_conversion_can_alias_numpy_source(
        device="cuda:0",
        source_dtype=source_dtype,
        target_dtype=None,
    )
