"""
Mask R-CNN int8 ONNX instance segmentation on a sample image.

Run from the repo root:
    python examples/run_maskrcnn.py
    python examples/run_maskrcnn.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
    MapTensor,
    Squeeze,
    TensorRegistry,
)
from ml_pipes.vision import (
    FilterTensorsByScore,
    ImagePayload,
    Normalize,
    ProjectBoxes,
    ProjectRoIMasks,
    Resize,
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


def build_inference_pipeline(model_path: Path) -> Pipeline[ImagePayload, TensorRegistry]:
    return Pipeline(
        [
            Resize((800, 800)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(
                scale=1.0,
                mean=_IMAGENET_MEAN_BGR,
                output_layout="CHW",
                output_color_space="BGR",
                add_batch_dim=False,
            ),
            Infer(model_path, input_layout="CHW", dtype="float32"),
            Extract("6568", "6570", "6572", "6887", as_=("boxes", "labels", "scores", "masks")),
            FilterTensorsByScore("boxes", "labels", "masks", score="scores", min_score=CONF_THRESHOLD),
            MapTensor("labels", fn=lambda t: t.astype(np.int32) - 1, as_="classes"),
            Recall("resize_transform"),
            ProjectBoxes(),  # model space → original image space
            Squeeze("masks", axis=1),  # (N, 1, 28, 28) → (N, 28, 28)
            Recall("resize_transform"),
            ProjectRoIMasks(),  # 28×28 RoI masks → full-image binary masks
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Mask R-CNN int8 ONNX instance segmentation demo on a COCO image.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to downloading the example Mask R-CNN int8 model.")
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
