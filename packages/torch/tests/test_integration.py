from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml_pipes.core import Pipeline
from ml_pipes.standard import Batch, Gather, Scatter, UnBatch
from ml_pipes.tensor import TensorPayload as NumpyTensorPayload
from ml_pipes.torch import (
    ToDevice,
    ToNumpy,
    ToNumpyRegistry,
    ToTorch,
    ToTorchRegistry,
    ArgMax,
    ApplyTensorMask,
    AsType,
    Collate,
    CreateTensorMask,
    CreateTensorMaskByThreshold,
    Distribute,
    Extract,
    GatherScores,
    Infer,
    MultiplyTensors,
    SelectTensors,
    Sigmoid,
    Slice,
    Softmax,
    SynchronizeTensors,
    TopK,
    TopKIndices2D,
)
from ml_pipes.torch.types import RuntimeOutputs, TensorPayload as TorchTensorPayload, TensorRegistry
from ml_pipes.validation import PipelineValidationError


class _TorchIdentity:
    def __call__(self, value: TorchTensorPayload) -> TorchTensorPayload:
        return value


class _TorchIncrementRegistry:
    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry["scores"] = registry["scores"] + 1
        return registry


class _TensorPayloadPassthrough:
    def __call__(self, value: NumpyTensorPayload) -> NumpyTensorPayload:
        return value


class _EmptyDetectionModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        device = x.device
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "class_logits": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "mask_logits": torch.zeros((0, 2, 2), dtype=torch.float32, device=device),
            "flat_scores": torch.zeros((0,), dtype=torch.float32, device=device),
        }


class _EmptyBatchedDetectionModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        device = x.device
        return {
            "boxes": torch.zeros((batch, 0, 4), dtype=torch.float32, device=device),
            "class_logits": torch.zeros((batch, 0, 3), dtype=torch.float32, device=device),
            "mask_logits": torch.zeros((batch, 0, 2, 2), dtype=torch.float32, device=device),
            "flat_scores": torch.zeros((batch, 0), dtype=torch.float32, device=device),
        }


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
        AsType("float16"),
        ToNumpy(),
    ])
    payload = NumpyTensorPayload(
        array=np.ones((1, 3, 4, 4), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

    assert isinstance(result, NumpyTensorPayload)
    assert result.layout == "NCHW"
    assert result.dtype == "float16"
    assert result.array.dtype == np.float16


def test_validate_mixed_domains_fail_without_explicit_conversion() -> None:
    pipeline = Pipeline([
        _TensorPayloadPassthrough(),
        Infer(torch.nn.Identity().eval()),
    ])

    with pytest.raises(PipelineValidationError, match="Infer"):
        pipeline.validate(inference=True)


def test_torch_infer_extract_and_registry_conversion_round_trip() -> None:
    class _Module(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "boxes": x + 1,
                "scores": x.sum(dim=1),
            }

    pipeline = Pipeline([
        ToTorch(),
        Infer(_Module().eval(), output_layouts=("NCHW", "NHW")),
        Extract("scores", as_="scores"),
        ToNumpyRegistry(),
    ])
    payload = NumpyTensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

    assert np.array_equal(result["scores"], np.full((1, 2, 2), 3.0, dtype=np.float32))


def test_torch_infer_and_distribute_preserve_empty_detection_axes_per_sample() -> None:
    pipeline = Pipeline([
        ToTorch(),
        Infer(
            _EmptyBatchedDetectionModule().eval(),
            output_layouts=("NNC", "NNC", "NNHW", "NN"),
        ),
        Distribute(),
    ])
    payload = NumpyTensorPayload(
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


def test_empty_torch_generic_postprocess_pipeline_keeps_empty_tensors_stable() -> None:
    pipeline = Pipeline([
        ToTorch(),
        ToDevice("cpu"),
        SynchronizeTensors(),
        AsType("float32"),
        Infer(
            _EmptyDetectionModule().eval(),
            output_layouts=("NC", "NC", "NHW", "N"),
        ),
        Extract("boxes", "class_logits", "mask_logits", "flat_scores"),
        Softmax("class_logits", as_="class_probs"),
        Slice("boxes", slice(None, 4), as_="boxes_xyxy"),
        Sigmoid("mask_logits", as_="mask_probs"),
        TopK("flat_scores", k=5, values_as="top_flat_scores", indices_as="top_flat_indices"),
        TopKIndices2D(
            "class_probs",
            k=5,
            values_as="top_scores",
            row_indices_as="query_indices",
            col_indices_as="class_ids",
        ),
        ArgMax("class_probs", as_="argmax_classes"),
        GatherScores("class_probs", "argmax_classes", as_="argmax_scores"),
        CreateTensorMask("top_scores", predicate=lambda tensor: tensor >= 0.0, as_="non_negative_keep"),
        CreateTensorMaskByThreshold("top_scores", threshold=0.5, as_="score_keep"),
        ApplyTensorMask("top_scores", "class_ids", "query_indices", mask="score_keep"),
        SelectTensors("mask_probs", indices="query_indices", as_="selected_masks"),
        MultiplyTensors("top_scores", "argmax_scores", as_="weighted_scores"),
        ToNumpyRegistry(),
    ])
    payload = NumpyTensorPayload(
        array=np.zeros((1, 3, 4, 4), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline(payload)

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


def test_torch_batch_region_validates_and_runs() -> None:
    pipeline = Pipeline([
        Batch(size=2, timeout=0.01),
        Collate(),
        Infer(torch.nn.Identity().eval()),
        Distribute(),
        UnBatch(),
    ])
    contract = pipeline.validate(inference=True)

    sample = _torch_payload(torch.ones((1, 3, 2, 2)))
    result = pipeline(sample)

    assert contract.input_type is TorchTensorPayload
    assert isinstance(result, RuntimeOutputs)
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
        Infer(torch.nn.Identity().eval(), serialize=True),
        Extract("output_0", as_="scores"),
    ])
    payload = NumpyTensorPayload(
        array=np.ones((1, 3, 2, 2), dtype=np.float32),
        layout="NCHW",
        dtype="float32",
    )

    result = pipeline.inspect(payload)

    spans = result.spans
    assert spans[0].output_shape == "TensorPayload (1, 3, 2, 2) @ cpu"
    assert spans[1].operator_config["device"] == "cpu"
    assert spans[2].output_shape == "RuntimeOutputs {output_0: (1, 3, 2, 2) @ cpu}"
    assert spans[3].output_shape == "TensorRegistry {scores: (1, 3, 2, 2) @ cpu}"


def test_torch_validation_accepts_to_device_and_torch_as_type_boundaries() -> None:
    contract = Pipeline([
        ToTorch(),
        ToDevice("cpu"),
        SynchronizeTensors(),
        AsType("float16"),
        _TorchIdentity(),
    ]).validate(inference=True)

    assert contract.input_type is NumpyTensorPayload
    assert contract.output_type is TorchTensorPayload


def test_torch_validation_rejects_registry_op_after_to_device_payload() -> None:
    with pytest.raises(PipelineValidationError):
        Pipeline([ToDevice("cpu"), ArgMax("scores")]).validate(
            pipeline_input_type=TorchTensorPayload
        )
