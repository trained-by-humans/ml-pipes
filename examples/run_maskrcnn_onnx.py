from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_pipes import (
    DecodeOp,
    InferOp,
    LogDetectionsOp,
    MapToObjectsOp,
    NormalizeOp,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    ResizeOp,
    ResizeTransform,
    Select,
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


def _project_rcnn_masks(registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
    """Resize each 28×28 RoI logit mask to its bounding box and embed into original-size canvas.

    Must be called AFTER ProjectBoxes — needs boxes already in original image space.
    """
    import cv2

    boxes = registry["boxes"]         # (N, 4) xyxy — original image space after ProjectBoxes
    masks_logits = registry["masks"]  # (N, 1, 28, 28) — raw logits from model
    orig_h, orig_w = transform.original_shape

    full_masks = np.zeros((len(boxes), orig_h, orig_w), dtype=bool)
    for i, (box, logits) in enumerate(zip(boxes, masks_logits)):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        mask = logits[0].astype(np.float32)
        resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        full_masks[i, y1:y2, x1:x2] = resized > 0.5

    registry["masks"] = full_masks
    return registry


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            DecodeOp(),
            ResizeOp((800, 800)),
            Store("resize_transform", index=1),
            Pick(0),
            NormalizeOp(
                scale=1.0,
                mean=_IMAGENET_MEAN_BGR,
                output_layout="CHW",
                output_color_space="BGR",
                add_batch_dim=False,
            ),
            InferOp(model_path, expected_input_layout="CHW", expected_model_dtype="float32"),
            Select("6568", "6570", "6572", "6887", as_=("boxes", "labels", "scores", "masks")),
            _filter_detections,
            Recall("resize_transform"),
            ProjectBoxes(),                # model space → original image space
            Recall("resize_transform"),
            _project_rcnn_masks,           # 28×28 RoI logits → full-image binary masks
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
            MapToObjectsOp(
                field_sources={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                    "mask_pixels": lambda value: [int(np.asarray(mask, dtype=np.uint8).sum()) for mask in value.masks],
                },
            ),
            LogDetectionsOp(
                model_path=model_path,
                image_path=image_path,
                annotated_image_path=output_path,
            ),
        ]
    )(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
