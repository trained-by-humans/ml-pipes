from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    Decode,
    GatherScores,
    Infer,
    LogDetections,
    MapToObjects,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    Resize,
    Scale,
    Select,
    Softmax,
    Squeeze,
    Store,
    ToDetections,
)
from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_detections,
)


MODEL_URL = "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx"
MODEL_NAME = "rfdetr_nano.onnx"

# RF-DETR exports normalized boxes in (cx, cy, w, h) format.
# Multiply by input size to get pixel coordinates.
INPUT_SIZE = (640, 640)


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            Decode(),
            Resize(target_size=INPUT_SIZE, mode="resize", interpolation="linear"),
            Store("resize_transform", index=1),
            Pick(0),
            Normalize(),
            Infer(model_path, dtype="float32"),
            Select("pred_boxes", "logits", as_=("boxes", "logits")),
            Squeeze("boxes"),                                   # (1, N, 4) → (N, 4)
            Squeeze("logits"),                                  # (1, N, C) → (N, C)
            Softmax("logits"),
            ArgMax("logits", as_="classes"),
            GatherScores("logits", "classes", as_="scores"),
            Scale("boxes", by=(INPUT_SIZE[1], INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[0])),  # normalized cxcywh → pixel cxcywh
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=0.25, iou_threshold=1.0, max_detections=20),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RF-DETR ONNX demo on a public COCO image.")
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
    render_and_save_detections(
        image_path=image_path,
        detections=result,
        output_path=output_path,
    )
    Pipeline(
        [
            MapToObjects(
                fields={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                }
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
