from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

from ..common import ASSETS_DIR, COCO_CLASSES, resolve_model_variant_path
from ..run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from .stream_common import FrameReader
from ml_pipes.core import (
    Embed,
    Pipeline,
)
from ml_pipes.standard import (
    Pick,
    Recall,
    Store,
)
from ml_pipes.vision import (
    DrawBoxes,
    ImagePayload,
)


def build_pipeline(model_path: Path) -> Pipeline[ImagePayload, ImagePayload]:
    return Pipeline([
        Store("source_frame"),
        Embed(yolo8_inference_pipeline(model_path)),
        Recall("source_frame", prepend=True),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0),
    ], auto_validate=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam inference with YOLOv8.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory used to cache downloaded models and sample assets.",
    )
    parser.add_argument(
        "--model",
        choices=list(YOLO8_MODELS),
        default="n",
        help=f"Model variant ({' → '.join(YOLO8_MODELS)}).",
    )
    args = parser.parse_args()

    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_variant_path(args.assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    pipeline = build_pipeline(model_path)
    pipeline.validate()
    pipeline.describe(verbose=True)

    try:
        reader = FrameReader(0)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Running — press Q in the window to quit.", file=sys.stderr)
    while True:
        ok, frame = reader.latest()
        if not ok:
            print("Warning: failed to read frame, stopping.", file=sys.stderr)
            break

        result = pipeline(ImagePayload(array=frame, color_space="BGR", layout="HWC"))
        cv2.imshow("YOLOv8 — Webcam", result.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
