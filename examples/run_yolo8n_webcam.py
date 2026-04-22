from __future__ import annotations

import sys
from pathlib import Path

import cv2

from common import COCO_CLASSES, download_if_missing
from run_yolo8n_onnx import MODEL_NAME, MODEL_URL, yolo8n_inference_pipeline
from ml_pipes import (
    DrawBoxes,
    ImagePayload,
    Pick,
    Pipeline,
    Recall,
    Store,
    Embed,
)

# Minimal live webcam inference with YOLOv8n.
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
        Embed(yolo8n_inference_pipeline(model_path)),
        Recall("source_frame", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0)
    ])


def main() -> int:
    model_path = ASSETS_DIR / MODEL_NAME
    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

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

        cv2.imshow("YOLOv8n — Webcam", annotated.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
