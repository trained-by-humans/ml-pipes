from __future__ import annotations

import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.tensor import (
    ApplyTensorMask,
    ArgMax,
    AsType,
    BinarizeTensorByThreshold,
    CreateTensorMask,
    CreateTensorMaskByThreshold,
    FilterTensors,
    GatherScores,
    MapTensor,
    MultiplyTensors,
    Scale,
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


def test_empty_tensor_registry_pipeline_keeps_empty_tensors_stable() -> None:
    pipeline = Pipeline([
        AsType("float16", src="class_logits_batched"),
        Squeeze("class_logits_batched", axis=1, as_="class_logits"),
        Scale("class_logits", by=1.0),
        Softmax("class_logits", as_="class_probs"),
        ArgMax("class_probs", as_="argmax_classes"),
        GatherScores("class_probs", "argmax_classes", as_="argmax_scores"),
        TopK("flat_scores", k=5, values_as="top_flat_scores", indices_as="top_flat_indices"),
        TopKIndices2D(
            "class_probs",
            k=5,
            values_as="scores",
            row_indices_as="row_indices",
            col_indices_as="class_ids",
        ),
        CreateTensorMaskByThreshold("scores", threshold=0.5, as_="score_keep"),
        CreateTensorMask("scores", predicate=lambda tensor: tensor >= 0.0, as_="non_negative_keep"),
        ApplyTensorMask("scores", "class_ids", "row_indices", mask="score_keep"),
        Sigmoid("mask_logits", as_="mask_probs"),
        MapTensor("mask_probs", fn=lambda tensor: tensor + 0.0, as_="mapped_mask_probs"),
        Transpose("mapped_mask_probs", axes=(0, 2, 1), as_="transposed_masks"),
        Slice("transposed_masks", at=slice(None, 2), as_="sliced_masks"),
        SelectTensors("sliced_masks", indices="row_indices", as_="selected_masks"),
        BinarizeTensorByThreshold("selected_masks", threshold=0.5, as_="binary_masks"),
        FilterTensors("selected_masks", "binary_masks", by="scores", predicate=lambda scores: scores >= 0.0),
        MultiplyTensors("scores", "argmax_scores", as_="weighted_scores"),
        SortTensorsBy("selected_masks", "binary_masks", "class_ids", "row_indices", by="weighted_scores"),
    ])
    registry = _registry(
        class_logits_batched=np.zeros((0, 1, 4), dtype=np.float32),
        mask_logits=np.zeros((0, 2, 3), dtype=np.float32),
        flat_scores=np.zeros((0,), dtype=np.float32),
    )

    result = pipeline(registry)

    assert result["class_logits_batched"].shape == (0, 1, 4)
    assert result["class_logits_batched"].dtype == np.float16
    assert result["class_logits"].shape == (0, 4)
    assert result["class_probs"].shape == (0, 4)
    assert result["argmax_classes"].shape == (0,)
    assert result["argmax_scores"].shape == (0,)
    assert result["top_flat_scores"].shape == (0,)
    assert result["top_flat_indices"].shape == (0,)
    assert result["scores"].shape == (0,)
    assert result["row_indices"].shape == (0,)
    assert result["class_ids"].shape == (0,)
    assert result["score_keep"].dtype == np.bool_
    assert result["non_negative_keep"].dtype == np.bool_
    assert result["mask_probs"].shape == (0, 2, 3)
    assert result["mapped_mask_probs"].shape == (0, 2, 3)
    assert result["transposed_masks"].shape == (0, 3, 2)
    assert result["sliced_masks"].shape == (0, 2, 2)
    assert result["selected_masks"].shape == (0, 2, 2)
    assert result["binary_masks"].shape == (0, 2, 2)
    assert result["binary_masks"].dtype == np.bool_
    assert result["weighted_scores"].shape == (0,)
