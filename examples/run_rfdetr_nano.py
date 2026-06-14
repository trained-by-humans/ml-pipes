from __future__ import annotations

import sys
from pathlib import Path

from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    download_if_missing,
    parse_input_and_output_args,
    visualize_detections_and_store,
)
from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    Detections,
    GatherScores,
    ImagePayload,
    Infer,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    Resize,
    Scale,
    Extract,
    Softmax,
    Squeeze,
    Store,
    ToDetections,
)

MODEL_URL = "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx"
MODEL_NAME = "rfdetr_nano.onnx"

# RF-DETR exports normalized boxes in (cx, cy, w, h) format.
# Multiply by input size to get pixel coordinates.
INPUT_SIZE = (640, 640)


def build_inference_pipeline(model_path: Path) -> Pipeline[ImagePayload, Detections]:
    return Pipeline(
        [
            Resize(target_size=INPUT_SIZE, mode="resize", interpolation="linear"),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(model_path, dtype="float32"),
            Extract("pred_boxes", "logits", as_=("boxes", "logits")),
            Squeeze("boxes"),  # (1, N, 4) → (N, 4)
            Squeeze("logits"),  # (1, N, C) → (N, C)
            Softmax("logits"),
            ArgMax("logits", as_="classes"),
            GatherScores("logits", "classes", as_="scores"),
            Scale("boxes", by=(INPUT_SIZE[1], INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[0])),
            # normalized cxcywh → pixel cxcywh
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=0.25, iou_threshold=1.0, max_detections=20),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
        ],
        auto_validate=True,
    )


def main() -> int:
    args = parse_input_and_output_args("Run an RF-DETR ONNX demo on a public COCO image.")
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, MODEL_NAME)

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    infer_pipe = build_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
