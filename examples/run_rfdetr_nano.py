"""
RF-DETR nano ONNX detection on a sample image.

Run from the repo root:
    python examples/run_rfdetr_nano.py
    python examples/run_rfdetr_nano.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    ASSETS_DIR,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    resolve_input_path,
    resolve_model_path,
    visualize_detections_and_store,
)
from ml_pipes.core import Pipeline
from ml_pipes.onnx import (
    Infer,
    Extract,
)
from ml_pipes.standard import (
    Pick,
    Recall,
    Store,
)
from ml_pipes.tensor import (
    ArgMax,
    GatherScores,
    Scale,
    Softmax,
    Squeeze,
)
from ml_pipes.vision import (
    ConvertBoxFormat,
    Detections,
    ImagePayload,
    NMS,
    Normalize,
    ProjectBoxes,
    Resize,
    ToDetections,
)

MODEL_URL = "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx"
MODEL_NAME = "rfdetr_nano.onnx"

# This detector exports normalized boxes in (cx, cy, w, h) format.
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
    parser = argparse.ArgumentParser(description="Run a DETR-style ONNX demo on a public COCO image.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to downloading the example RF-DETR nano model.")
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to a file under the assets directory.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, ASSETS_DIR / MODEL_NAME, MODEL_URL)
    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, model_path.name)

    infer_pipe = build_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
