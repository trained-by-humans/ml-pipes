from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_pipes import (
    DecodeOp,
    DecodeSegmentationOp,
    InferOp,
    LogDetectionsOp,
    MapToObjectsOp,
    NormalizeOp,
    Pipeline,
    ProjectSegmentationsOp,
    Recall,
    ResizeOp,
    SegmentationNMSOp,
    Select,
    Store,
)
from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_segmentations,
)


MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx"
MODEL_NAME = "yolo11n-seg.onnx"


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            DecodeOp(),
            ResizeOp((640, 640)),
            Store("resize_transform", index=1),
            Select(0),
            NormalizeOp(),
            InferOp(model_path, expected_input_layout="NCHW", expected_model_dtype="float32"),
            DecodeSegmentationOp(
                export_detection_output_index=0,
                export_prototype_output_index=1,
                num_masks=32,
                input_box_format="xywh",
                transpose_output="auto",
                squeeze_batch_dim=True,
            ),
            SegmentationNMSOp(conf_threshold=0.25, iou_threshold=0.45, max_detections=100),
            Recall("resize_transform"),
            ProjectSegmentationsOp(mask_threshold=0.5),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLO11n-seg ONNX instance segmentation demo on a COCO image.")
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
