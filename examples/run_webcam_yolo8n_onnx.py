from __future__ import annotations

import sys
from pathlib import Path

import cv2

from common import COCO_CLASSES, download_if_missing
from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    DrawBoxes,
    GatherScores,
    ImagePayload,
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

# Minimal live webcam inference with YOLOv8n.
#
# Reads frames from the default camera, runs detection on each frame, and
# displays the result in a window.  Press Q to quit.
#
# The pipeline starts at Resize rather than Decode because cv2.VideoCapture
# already gives us a decoded BGR array — there is no file path to read.

MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MODEL_NAME = "yolov8n.onnx"
ASSETS_DIR = Path(".example_assets")


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Resize((640, 640)),
        Store("resize_transform", index=1),
        Pick(0),
        Normalize(),
        Infer(model_path),
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
    ])


def main() -> int:
    model_path = ASSETS_DIR / MODEL_NAME
    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    pipeline = build_pipeline(model_path)
    draw = DrawBoxes(class_names=COCO_CLASSES)

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
        detections = pipeline(source)
        annotated, _ = draw(source, detections)

        cv2.imshow("YOLOv8n — Webcam", annotated.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
