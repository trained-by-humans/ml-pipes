from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from common import COCO_CLASSES, add_model_arg, resolve_model_path
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import (
    DrawBoxes,
    ImagePayload,
    Pick,
    Pipeline,
    Recall,
    Store,
    Embed,
)

# Minimal live webcam inference with YOLOv8.
#
# Reads frames from the default camera, runs detection on each frame, and
# displays the result in a window.  Press Q to quit.
#
# The pipeline starts at Resize rather than Decode because cv2.VideoCapture
# already gives us a decoded BGR array — there is no file path to read.

ASSETS_DIR = Path(".example_assets")


def build_webcam_annotation_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Store("source_frame"),
        Embed(yolo8_inference_pipeline(model_path)),
        Recall("source_frame", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0)
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam inference with YOLOv8.")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    add_model_arg(parser, list(YOLO8_MODELS))
    args = parser.parse_args()

    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_path(args.assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    pipeline = build_webcam_annotation_pipeline(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.", file=sys.stderr)
        return 1

    print("Running — press Q in the window to quit.", file=sys.stderr)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame, stopping.", file=sys.stderr)
            break

        source = ImagePayload(array=frame, color_space="BGR", layout="HWC")
        annotated = pipeline(source)

        cv2.imshow("YOLOv8 — Webcam", annotated.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
