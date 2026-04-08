from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_pipes import (
    DecodeOp,
    DecodePredictionsOp,
    InferOp,
    NMSOp,
    NormalizeOp,
    Pipeline,
    ProjectToInputOp,
    Recall,
    ResizeOp,
    Select,
    Store,
)
from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_detections,
)


MODEL_URL = "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n_fp16.onnx"
MODEL_NAME = "yolo11n_fp16.onnx"


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            DecodeOp(),
            ResizeOp(
                size=(640, 640),
                mode="letterbox",
                pad_value=114,
                interpolation="linear",
                center=True,
                allow_scale_up=True,
            ),
            Store("resize_transform", index=1),
            Select(0),
            NormalizeOp(
                output_dtype="float16",
                scale=1.0 / 255.0,
                output_layout="NCHW",
                output_color_space="RGB",
                add_batch_dim=True,
            ),
            InferOp(
                model_path,
                expected_input_layout="NCHW"
            ),
            DecodePredictionsOp(
                num_box_values=4,
                class_start_index=4,
                input_box_format="xywh",
                transpose_output="auto",
                squeeze_batch_dim=True,
                score_activation="none",
            ),
            NMSOp(),
            Recall("resize_transform"),
            ProjectToInputOp(),
        ]
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLO11 FP16 ONNX demo on a COCO image.")
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
        class_names=COCO_CLASSES,
    )

    detections = [
        {
            "box": box,
            "score": score,
            "class_id": class_id,
            "label": COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id),
        }
        for box, score, class_id in zip(result.boxes, result.scores, result.classes, strict=True)
    ]
    print(
        json.dumps(
            {
                "image": str(image_path),
                "model": str(model_path),
                "annotated_image": str(output_path),
                "detections": detections,
                "config": {
                    "normalize_dtype": "float16",
                    "resize_mode": "letterbox",
                    "decoder_transpose": "auto",
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
