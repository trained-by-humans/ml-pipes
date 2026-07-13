from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes.core import Pipeline
from ml_pipes.standard import Batch, Gather, Pick, Recall, Scatter, Store, UnBatch
from ml_pipes.tensor import TensorPayload
from ml_pipes.torch import (
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
    TorchArgMax,
    TorchApplyTensorMask,
    TorchAsType,
    TorchBinarizeTensorByThreshold,
    TorchCollate,
    TorchCreateTensorMask,
    TorchCreateTensorMaskByThreshold,
    TorchDistribute,
    TorchExtract,
    TorchFilterTensorsByClasses,
    TorchFilterTensorsByMasksArea,
    TorchFilterTensorsByScore,
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
from ml_pipes.torch.types import TorchRuntimeOutputs, TorchTensorPayload, TorchTensorRegistry
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vision import Normalize


class _TorchIdentity:
    def __call__(self, value: TorchTensorPayload) -> TorchTensorPayload:
        return value


class _TorchIncrementRegistry:
    def __call__(self, registry: TorchTensorRegistry) -> TorchTensorRegistry:
        registry["scores"] = registry["scores"] + 1
        return registry


class _EmptyDetectionModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 2, 2), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )


class _EmptyBatchedDetectionModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = int(x.shape[0])
        device = x.device
        return (
            torch.zeros((batch, 0, 4), dtype=torch.float32, device=device),
            torch.zeros((batch, 0, 3), dtype=torch.float32, device=device),
            torch.zeros((batch, 0, 2, 2), dtype=torch.float32, device=device),
            torch.zeros((batch, 0), dtype=torch.float32, device=device),
        )


def _torch_payload(array: torch.Tensor, layout: str = "NCHW") -> TorchTensorPayload:
    return TorchTensorPayload(
        array=array,
        layout=layout,
        dtype=str(array.dtype).replace("torch.", ""),
        device=str(array.device),
    )


def test_numpy_torch_numpy_pipeline_composes() -> None:
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


def test_validate_mixed_domains_fail_without_explicit_conversion() -> None:
    pipeline = Pipeline([
        Normalize(),
        TorchInfer(torch.nn.Identity().eval()),
    ])

    with pytest.raises(PipelineValidationError, match="TorchInfer"):
        pipeline.validate(inference=True)


def test_torch_infer_extract_and_registry_conversion_round_trip() -> None:
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


def test_torch_infer_and_distribute_preserve_empty_detection_axes_per_sample() -> None:
    pipeline = Pipeline([
        ToTorch(),
        TorchInfer(
            _EmptyBatchedDetectionModule().eval(),
            output_names=("boxes", "class_logits", "mask_logits", "flat_scores"),
            output_layouts=("NNC", "NNC", "NNHW", "NN"),
        ),
        TorchDistribute(),
    ])
    payload = TensorPayload(
        array=np.zeros((2, 3, 4, 4), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

    assert len(result) == 2
    for sample in result:
        assert sample.names == ("boxes", "class_logits", "mask_logits", "flat_scores")
        assert tuple(sample.tensors[0].array.shape) == (1, 0, 4)
        assert tuple(sample.tensors[1].array.shape) == (1, 0, 3)
        assert tuple(sample.tensors[2].array.shape) == (1, 0, 2, 2)
        assert tuple(sample.tensors[3].array.shape) == (1, 0)


def test_empty_torch_postprocess_pipeline_keeps_empty_tensors_stable() -> None:
    pipeline = Pipeline([
        Store("image_shape", source=1),
        Pick(0),
        ToTorch(),
        ToDevice("cpu"),
        TorchSynchronizeTensors(),
        TorchAsType("float32"),
        TorchInfer(
            _EmptyDetectionModule().eval(),
            output_names=("boxes", "class_logits", "mask_logits", "flat_scores"),
            output_layouts=("NC", "NC", "NHW", "N"),
        ),
        TorchExtract("boxes", "class_logits", "mask_logits", "flat_scores"),
        TorchSoftmax("class_logits", as_="class_probs"),
        TorchSlice("boxes", slice(None, 4), as_="boxes_xyxy"),
        TorchSigmoid("mask_logits", as_="mask_probs"),
        TorchTopK("flat_scores", k=5, values_as="top_flat_scores", indices_as="top_flat_indices"),
        TorchTopKIndices2D(
            "class_probs",
            k=5,
            values_as="top_scores",
            row_indices_as="query_indices",
            col_indices_as="class_ids",
        ),
        TorchArgMax("class_probs", as_="argmax_classes"),
        TorchGatherScores("class_probs", "argmax_classes", as_="argmax_scores"),
        TorchCreateTensorMask("top_scores", predicate=lambda tensor: tensor >= 0.0, as_="non_negative_keep"),
        TorchCreateTensorMaskByThreshold("top_scores", threshold=0.5, as_="score_keep"),
        TorchApplyTensorMask("top_scores", "class_ids", "query_indices", mask="score_keep"),
        TorchSelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
        TorchMultiplyTensors("top_scores", "argmax_scores", as_="weighted_scores"),
        TorchFilterTensorsByScore("selected_masks", "class_ids", "query_indices", score="weighted_scores", min_score=0.0),
        TorchFilterTensorsByClasses(
            "selected_masks",
            "weighted_scores",
            "query_indices",
            classes="class_ids",
            keep_classes={0, 1, 2},
        ),
        TorchWeightMasksByScores(masks="selected_masks", scores="weighted_scores", as_="weighted_masks"),
        Recall("image_shape"),
        TorchResizeMasks(masks="weighted_masks", as_="resized_masks"),
        TorchMeanMaskScores(masks="resized_masks", binary_masks=None, as_="mean_mask_scores"),
        TorchBinarizeTensorByThreshold("resized_masks", threshold=0.5, as_="binary_masks"),
        TorchFilterTensorsByMasksArea("weighted_scores", "class_ids", masks="binary_masks", min_area=1),
        TorchMasksToBoxes(masks="binary_masks", as_="boxes"),
        TorchNMS(boxes="boxes", scores="weighted_scores", classes="class_ids", kept_as="kept"),
        TorchSortTensorsBy("binary_masks", "class_ids", by="weighted_scores"),
        ToNumpyRegistry(),
    ])
    payload = TensorPayload(
        array=np.zeros((1, 3, 4, 4), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline((payload, (4, 5)))

    assert result["boxes"].shape == (0, 4)
    assert result["boxes"].dtype == np.float32
    assert result["class_logits"].shape == (0, 3)
    assert result["class_probs"].shape == (0, 3)
    assert result["boxes_xyxy"].shape == (0, 4)
    assert result["mask_probs"].shape == (0, 2, 2)
    assert result["top_flat_scores"].shape == (0,)
    assert result["top_flat_indices"].shape == (0,)
    assert result["top_scores"].shape == (0,)
    assert result["query_indices"].shape == (0,)
    assert result["class_ids"].shape == (0,)
    assert result["argmax_classes"].shape == (0,)
    assert result["argmax_scores"].shape == (0,)
    assert result["non_negative_keep"].dtype == np.bool_
    assert result["score_keep"].dtype == np.bool_
    assert result["selected_masks"].shape == (0, 2, 2)
    assert result["weighted_scores"].shape == (0,)
    assert result["weighted_masks"].shape == (0, 2, 2)
    assert result["resized_masks"].shape == (0, 4, 5)
    assert result["mean_mask_scores"].shape == (0,)
    assert result["binary_masks"].shape == (0, 4, 5)
    assert result["binary_masks"].dtype == np.bool_
    assert result["kept"].shape == (0,)


def test_torch_batch_region_validates_and_runs() -> None:
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


def test_torch_scatter_region_validates_and_runs() -> None:
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


def test_torch_registry_conversion_handoff_back_to_numpy() -> None:
    pipeline = Pipeline([
        ToTorchRegistry(),
        _TorchIncrementRegistry(),
        ToNumpyRegistry(),
    ])
    from ml_pipes.tensor import TensorRegistry

    registry = TensorRegistry({"scores": np.array([1.0, 2.0], dtype=np.float32)})
    result = pipeline(registry)

    assert np.array_equal(result["scores"], np.array([2.0, 3.0], dtype=np.float32))


def test_torch_inspection_captures_device_shapes_and_operator_config() -> None:
    pipeline = Pipeline([
        ToTorch(device="cpu"),
        ToDevice("cpu"),
        TorchInfer(torch.nn.Identity().eval(), serialize=True),
        TorchExtract("output_0", as_="scores"),
    ])
    payload = TensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline.inspect(payload)

    spans = result.spans
    assert spans[0].output_shape == "TorchTensorPayload (1, 3, 2, 2) @ cpu"
    assert spans[1].operator_config["device"] == "cpu"
    assert spans[2].output_shape == "TorchRuntimeOutputs {output_0: (1, 3, 2, 2) @ cpu}"
    assert spans[3].output_shape == "TorchTensorRegistry {scores: (1, 3, 2, 2) @ cpu}"


def test_torch_validation_accepts_to_device_and_torch_as_type_boundaries() -> None:
    contract = Pipeline([
        ToTorch(),
        ToDevice("cpu"),
        TorchSynchronizeTensors(),
        TorchAsType("float16"),
        _TorchIdentity(),
    ]).validate(inference=True)

    assert contract.input_type is TensorPayload
    assert contract.output_type is TorchTensorPayload


def test_torch_validation_rejects_registry_op_after_to_device_payload() -> None:
    with pytest.raises(PipelineValidationError):
        Pipeline([ToDevice("cpu"), TorchArgMax("scores")]).validate(
            pipeline_input_type=TorchTensorPayload
        )
