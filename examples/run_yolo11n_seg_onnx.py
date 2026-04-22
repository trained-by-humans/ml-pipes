from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    Decode,
    LoadFile,
    GatherScores,
    Infer,
    LogDetections,
    MapToObjects,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    FilterBy,
    ProjectBoxes,
    ProjectMasks,
    ReconstructMasks,
    Recall,
    Resize,
    Extract,
    Slice,
    Squeeze,
    Store,
    ToSegmentations,
    Transpose,
)
from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_segmentations,
)


MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.onnx"
MODEL_NAME = "yolo11n-seg.onnx"

# output0: (1, 116, 8400) — 4 box + 80 class + 32 mask coefficients
# output1: (1, 32, 160, 160) — prototype masks
NUM_MASKS = 32


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            Resize((640, 640)),
            Store("resize_transform", index=1),
            Pick(0),
            Normalize(),
            Infer(model_path, dtype="float32"),
            Extract("output0", "output1", as_=("preds", "protos")),
            Squeeze("preds"),                                              # (1, 116, N) → (116, N)
            Squeeze("protos"),                                             # (1, 32, H, W) → (32, H, W)
            Transpose("preds"),                                            # (116, N) → (N, 116)
            Slice("preds", slice(None, 4), as_="boxes"),                  # (N, 4)
            Slice("preds", slice(4, -NUM_MASKS), as_="class_scores"),     # (N, 80)
            Slice("preds", slice(-NUM_MASKS, None), as_="mask_coeffs"),   # (N, 32)
            ArgMax("class_scores", as_="classes"),
            GatherScores("class_scores", "classes", as_="scores"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(kept_as="kept"),
            FilterBy("mask_coeffs", "kept"),
            ReconstructMasks("mask_coeffs", "protos", as_="masks"),
            Recall("resize_transform"),
            ProjectBoxes(),
            Recall("resize_transform"),
            ProjectMasks(),
            ToSegmentations(),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a YOLO11n-seg ONNX instance segmentation demo on a COCO image.")
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
    render_and_save_segmentations(
        image_path=image_path,
        segmentations=result,
        output_path=output_path,
        class_names=COCO_CLASSES,
    )
    Pipeline(
        [
            MapToObjects(
                fields={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                    "mask_pixels": lambda value: [int(np.asarray(mask, dtype=np.uint8).sum()) for mask in value.masks],
                },
            ),
            LogDetections(
                model_path=model_path,
                image_path=image_path,
                annotated_image_path=output_path,
            ),
        ]
    )(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
