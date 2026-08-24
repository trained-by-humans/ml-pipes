"""
YOLO11n FP16 ONNX detection on a sample image.

Run from the repo root:
    python examples/run_yolo11n_fp16.py
    python examples/run_yolo11n_fp16.py --input path/to/photo.jpg
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
    AsType,
    ArgMax,
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
    Resize,
)

MODEL_URL = "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n_fp16.onnx"
MODEL_NAME = "yolo11n_fp16.onnx"


def build_inference_pipeline(model_path: Path) -> Pipeline[ImagePayload, TensorRegistry]:
    return Pipeline(
        [
            Resize(
                target_size=(640, 640),
                mode="letterbox",
                pad_value=114,
                interpolation="linear",
                center=True,
                allow_scale_up=True,
            ),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            AsType("float16"),
            Infer(model_path, dtype="float16"),
            Extract("output0", as_="preds"),
            AsType(src="preds", dtype="float32"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(),
            Recall("resize_transform"),
            ProjectBoxes(),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO11 FP16 ONNX demo on a COCO image.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to downloading the example YOLO11n FP16 model.")
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to a file under the assets directory.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, ASSETS_DIR / MODEL_NAME, MODEL_URL)
    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, model_path.name)

    infer_pipe = build_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
