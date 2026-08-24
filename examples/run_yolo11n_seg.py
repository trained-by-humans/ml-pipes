"""
YOLO11n-seg ONNX instance segmentation on a sample image.

Run from the repo root:
    python examples/run_yolo11n_seg.py
    python examples/run_yolo11n_seg.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    ASSETS_DIR,
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    resolve_input_path,
    resolve_model_path,
    visualize_and_store,
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
    SelectTensors,
    GatherScores,
    Slice,
    Squeeze,
    Transpose,
    TensorRegistry,
)
from ml_pipes.vision import (
    ConvertBoxFormat,
    ImagePayload,
    NMS,
    Normalize,
    ProjectBoxes,
    ProjectMasks,
    ReconstructMasks,
    Resize,
)

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx"
MODEL_NAME = "yolo11n-seg.onnx"

# output0: (1, 116, 8400) — 4 box + 80 class + 32 mask coefficients
# output1: (1, 32, 160, 160) — prototype masks
NUM_MASKS = 32


def build_inference_pipeline(model_path: Path) -> Pipeline[ImagePayload, TensorRegistry]:
    return Pipeline(
        [
            Resize((640, 640)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(model_path, dtype="float32"),
            Extract("output0", "output1", as_=("preds", "protos")),
            Squeeze("preds"),  # (1, 116, N) → (116, N)
            Squeeze("protos"),  # (1, 32, H, W) → (32, H, W)
            Transpose("preds"),  # (116, N) → (N, 116)
            Slice("preds", slice(None, 4), as_="boxes"),  # (N, 4)
            Slice("preds", slice(4, -NUM_MASKS), as_="class_scores"),  # (N, 80)
            Slice("preds", slice(-NUM_MASKS, None), as_="mask_coeffs"),  # (N, 32)
            ArgMax("class_scores", as_="classes"),
            GatherScores("class_scores", "classes", as_="scores"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(kept_as="kept"),
            SelectTensors("mask_coeffs", indices="kept"),
            ReconstructMasks("mask_coeffs", "protos", as_="masks"),
            Recall("resize_transform"),
            ProjectBoxes(),
            Recall("resize_transform"),
            ProjectMasks(),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO11n-seg ONNX instance segmentation demo on a COCO image.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to downloading the example YOLO11n-seg model.")
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to a file under the assets directory.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, ASSETS_DIR / MODEL_NAME, MODEL_URL)
    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, model_path.name)

    infer_pipe = build_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_and_store(output_path, COCO_CLASSES)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
