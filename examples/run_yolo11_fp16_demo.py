from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from ml_pipes import (
    DecodeOp,
    DecodePredictionsOp,
    DrawBoxesOp,
    InferOp,
    NMSOp,
    NormalizeOp,
    Pipeline,
    ProjectToInputOp,
    Recall,
    ResizeOp,
    SaveImageOp,
    Select,
    Store,
)


MODEL_URL = "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n_fp16.onnx"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

MODEL_NAME = "yolo11n_fp16.onnx"
IMAGE_NAME = "coco_000000039769.jpg"

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


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


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


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
        default=Path(".example_assets/coco_000000039769_yolo11_fp16_annotated.jpg"),
        help="Where to save the annotated image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / IMAGE_NAME

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(IMAGE_URL, image_path)

    source_image = DecodeOp()(image_path)
    pipeline = build_pipeline(model_path)
    result = pipeline(image_path)
    Pipeline(
        [
            lambda detections: (detections, source_image.array),
            DrawBoxesOp(class_names=COCO_CLASSES),
            SaveImageOp(args.output),
        ]
    )(result)

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
                "annotated_image": str(args.output),
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
