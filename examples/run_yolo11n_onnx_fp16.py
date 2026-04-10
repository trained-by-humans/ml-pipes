from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_pipes import (
    ArgMax,
    CastTensorOp,
    ConvertBoxFormat,
    DecodeOp,
    GatherScores,
    InferOp,
    LogDetectionsOp,
    MapToObjectsOp,
    NMS,
    NormalizeOp,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    ResizeOp,
    Select,
    Slice,
    Squeeze,
    Store,
    ToDetections,
    Transpose,
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
                target_size=(640, 640),
                mode="letterbox",
                pad_value=114,
                interpolation="linear",
                center=True,
                allow_scale_up=True,
            ),
            Store("resize_transform", index=1),
            Pick(0),
            NormalizeOp(
                scale=1.0 / 255.0,
                output_layout="NCHW",
                output_color_space="RGB",
                add_batch_dim=True,
            ),
            CastTensorOp("float16"),
            InferOp(
                model_path,
                expected_input_layout="NCHW",
                expected_model_dtype="float16",
            ),
            CastTensorOp("float32", selector="tensors"),
            Select("output0", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat("boxes", from_="cxcywh", to="xyxy"),
            NMS(),
            Recall("resize_transform"),
            ProjectBoxes(),
            ToDetections(),
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
    Pipeline(
        [
            MapToObjectsOp(
                field_sources={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                },
            ),
            LogDetectionsOp(
                model_path=model_path,
                image_path=image_path,
                annotated_image_path=output_path,
            ),
        ]
    )(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
