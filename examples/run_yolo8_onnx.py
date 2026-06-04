from __future__ import annotations

import sys
from pathlib import Path

import argparse

from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_model_arg,
    build_output_path,
    decode,
    download_if_missing,
    resolve_model_path,
    visualize_detections_and_store,
)
from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    GatherScores,
    Infer,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    Resize,
    Extract,
    Slice,
    Squeeze,
    Store,
    ToDetections,
    Transpose,
)

# Registry of all supported YOLOv8 variants.
# n auto-downloads; s/m/l/x must be exported locally:
#   yolo export model=yolov8{s,m,l,x}.pt format=onnx
YOLO8_MODELS: dict[str, tuple[str, str | None]] = {
    "n": ("yolov8n.onnx", "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"),
    "s": ("yolov8s.onnx", None),
    "m": ("yolov8m.onnx", None),
    "l": ("yolov8l.onnx", None),
    "x": ("yolov8x.onnx", None),
}


def yolo8_inference_pipeline(model_path: Path, conf_threshold: float = 0.25) -> Pipeline:
    return Pipeline(
        [
            Resize((640, 640)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(model_path),
            Extract("output0", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=conf_threshold),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 ONNX detection on a COCO image.")
    add_assets_dir_arg(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    add_model_arg(parser, list(YOLO8_MODELS))
    args = parser.parse_args()

    assets_dir = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    if args.input is not None:
        image_path = args.input
        if not image_path.exists():
            print(f"Error: input file not found: {image_path}", file=sys.stderr)
            return 1
    else:
        image_path = assets_dir / COCO_IMAGE_NAME
        print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
        download_if_missing(COCO_IMAGE_URL, image_path)

    output_path = args.output or build_output_path(assets_dir, image_path.name, model_name)

    infer_pipe = yolo8_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
