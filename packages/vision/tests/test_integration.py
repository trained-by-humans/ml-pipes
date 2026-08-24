from __future__ import annotations

import io

import numpy as np

from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Recall, Store
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import (
    ConvertBoxFormat,
    DrawBoxes,
    DrawMasks,
    FilterTensorsByClasses,
    FilterTensorsByMasksArea,
    FilterTensorsByScore,
    ImagePayload,
    LogDetections,
    MasksToBoxes,
    MeanMaskScores,
    NMM,
    NMS,
    ProjectBoxes,
    ProjectMasks,
    ProjectRoIMasks,
    ReconstructMasks,
    ResizeMasks,
    ResizeTransform,
    SaveImage,
    WeightMasksByScores,
)


def test_empty_detection_pipeline_preserves_empty_outputs(tmp_path) -> None:
    stream = io.StringIO()
    image = ImagePayload(array=np.zeros((8, 10, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    transform = ResizeTransform(
        scale=(1.0, 1.0),
        pad=(0.0, 0.0),
        original_shape=image.spatial_shape,
        resized_shape=image.spatial_shape,
    )
    registry = TensorRegistry(
        {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
        }
    )
    output_path = tmp_path / "detections.jpg"
    pipeline = Pipeline([
        Store("resize_transform", source=1),
        Store("source_image", source=2),
        Pick(0),
        ConvertBoxFormat(from_="xyxy"),
        FilterTensorsByScore("boxes", "classes", score="scores", min_score=0.5),
        FilterTensorsByClasses("boxes", "scores", classes="classes", keep_classes={0, 1}),
        NMS(kept_as="kept"),
        Recall("resize_transform"),
        ProjectBoxes(),
        NMM(iou_threshold=0.5),
        Recall("source_image", prepend=True),
        DrawBoxes(class_names=["zero", "one"]),
        LogDetections(
            model_path="model.onnx",
            image_path="image.jpg",
            annotated_image_path=output_path,
            stream=stream,
            at=1,
        ),
        SaveImage(output_path, at=0),
    ])

    result = pipeline((registry, transform, image))

    assert output_path.is_file()
    assert isinstance(result[0], ImagePayload)
    assert isinstance(result[1], TensorRegistry)
    assert result[1]["boxes"].shape == (0, 4)
    assert np.array_equal(result[0].array, image.array)
    assert '"detections": []' in stream.getvalue()


def test_empty_segmentation_pipeline_preserves_empty_outputs(tmp_path) -> None:
    stream = io.StringIO()
    image = ImagePayload(array=np.zeros((4, 5, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
    transform = ResizeTransform(
        scale=(1.0, 1.0),
        pad=(0.0, 0.0),
        original_shape=image.spatial_shape,
        resized_shape=image.spatial_shape,
    )
    registry = TensorRegistry(
        {
            "coefficients": np.zeros((0, 3), dtype=np.float32),
            "prototypes": np.ones((3, 2, 2), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
        }
    )
    output_path = tmp_path / "segmentations.jpg"
    pipeline = Pipeline([
        Store("resize_transform", source=1),
        Store("image_shape", source=2),
        Store("source_image", source=3),
        Pick(0),
        ReconstructMasks("coefficients", "prototypes", as_="masks"),
        WeightMasksByScores(masks="masks", scores="scores", as_="weighted_masks"),
        Recall("image_shape"),
        ResizeMasks(masks="weighted_masks", as_="resized_masks"),
        MeanMaskScores(masks="resized_masks", binary_masks=None, as_="mean_mask_scores"),
        FilterTensorsByMasksArea("scores", "classes", masks="resized_masks", min_area=1),
        FilterTensorsByScore("resized_masks", "classes", score="scores", min_score=0.5),
        MasksToBoxes(masks="resized_masks", as_="boxes"),
        Recall("resize_transform"),
        ProjectBoxes(),
        Recall("resize_transform"),
        ProjectMasks(masks="weighted_masks", boxes="boxes", mask_threshold=0.0),
        Recall("resize_transform"),
        ProjectRoIMasks(masks="masks", boxes="boxes", mask_threshold=0.0),
        Recall("source_image", prepend=True),
        DrawMasks(alpha=0.6),
        LogDetections(
            model_path="model.onnx",
            image_path="image.jpg",
            annotated_image_path=output_path,
            stream=stream,
            at=1,
        ),
        SaveImage(output_path, at=0),
    ])

    result = pipeline((registry, transform, image.spatial_shape, image))

    assert output_path.is_file()
    assert isinstance(result[0], ImagePayload)
    assert isinstance(result[1], TensorRegistry)
    assert result[1]["masks"].shape[0] == 0
    assert np.array_equal(result[0].array, image.array)
    assert '"detections": []' in stream.getvalue()
