from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_pipes import (
    Decode,
    Infer,
    LogDetections,
    MapToObjects,
    Normalize,
    Pick,
    Pipeline,
    ProjectBoxes,
    ProjectRoIMasks,
    Recall,
    Resize,
    Select,
    Squeeze,
    Store,
    TensorRegistry,
    ToSegmentations,
)
from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_segmentations,
)


MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation"
    "/mask-rcnn/model/MaskRCNN-12-int8.onnx"
)
MODEL_NAME = "MaskRCNN-12-int8.onnx"

# Input contract: float32, BGR, CHW (no batch dim), mean-subtracted with ImageNet BGR means.
# NMS is baked into the model — outputs are already filtered detections.
_IMAGENET_MEAN_BGR = (102.9801, 115.9465, 122.7717)
CONF_THRESHOLD = 0.7


def _filter_detections(registry: TensorRegistry) -> TensorRegistry:
    """Keep detections above confidence threshold; convert 1-indexed COCO labels to 0-indexed."""
    kept = np.where(registry["scores"] >= CONF_THRESHOLD)[0]
    registry["boxes"] = registry["boxes"][kept]
    registry["scores"] = registry["scores"][kept]
    registry["masks"] = registry["masks"][kept]
    registry["classes"] = registry["labels"][kept].astype(np.int32) - 1  # COCO 1-indexed → 0-indexed
    return registry


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            Decode(),
            Resize((800, 800)),
            Store("resize_transform", index=1),
            Pick(0),
            Normalize(
                scale=1.0,
                mean=_IMAGENET_MEAN_BGR,
                output_layout="CHW",
                output_color_space="BGR",
                add_batch_dim=False,
            ),
            Infer(model_path, expected_input_layout="CHW", expected_model_dtype="float32"),
            Select("6568", "6570", "6572", "6887", as_=("boxes", "labels", "scores", "masks")),
            _filter_detections,
            Recall("resize_transform"),
            ProjectBoxes(),                        # model space → original image space
            Squeeze("masks", axis=1),              # (N, 1, 28, 28) → (N, 28, 28)
            Recall("resize_transform"),
            ProjectRoIMasks(mask_threshold=0.5),   # 28×28 RoI masks → full-image binary masks
            ToSegmentations(),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Mask R-CNN int8 ONNX instance segmentation demo on a COCO image.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(".example_assets"),
        help="Directory used to cache the downloaded public model and image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the annotated image. Defaults to the input image name with the model name as suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, MODEL_NAME)

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    pipeline = build_pipeline(model_path)
    result = pipeline(image_path)
    render_and_save_segmentations(
        image_path=image_path,
        segmentations=result,
        output_path=output_path,
        class_names=COCO_CLASSES,
    )
    Pipeline(
        [
            MapToObjects(
                field_sources={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                    "mask_pixels": lambda value: [int(np.asarray(mask, dtype=np.uint8).sum()) for mask in value.masks],
                },
            ),
            LogDetections(
                model_path=model_path,
                image_path=image_path,
                annotated_image_path=output_path,
            ),
        ]
    )(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
