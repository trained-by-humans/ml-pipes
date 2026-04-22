from __future__ import annotations

import sys
from pathlib import Path

from common import (
    COCO_CLASSES,
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
    Cast,
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

MODEL_URL = "https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n_fp16.onnx"
MODEL_NAME = "yolo11n_fp16.onnx"


def build_inference_pipeline(model_path: Path) -> Pipeline:
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
            Store("resize_transform", index=1),
            Pick(0),
            Normalize(),
            Cast("float16"),
            Infer(model_path, dtype="float16"),
            Cast("float32", field="tensors"),
            Extract("output0", as_="preds"),
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
            ToDetections(),
        ]
    )


def main() -> int:
    args = parse_input_and_output_args("Run a YOLO11 FP16 ONNX demo on a COCO image.")
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, MODEL_NAME)

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    infer_pipe = build_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
