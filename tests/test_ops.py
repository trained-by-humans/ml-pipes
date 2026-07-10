from __future__ import annotations

import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.tensor import (
    BinarizeTensorByThreshold,
    MultiplyTensors,
    SelectTensors,
    SortTensorsBy,
    TensorRegistry,
    TopKIndices2D,
)
from ml_pipes.vision import (
    FilterTensorsByMasksArea,
    FilterTensorsByScore,
    MasksToBoxes,
    MeanMaskScores,
    Segmentations,
    ToSegmentations,
)


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
