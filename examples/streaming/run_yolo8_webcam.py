"""
Live YOLOv8 webcam detection.

Run from the repo root:
    python examples/streaming/run_yolo8_webcam.py
    python examples/streaming/run_yolo8_webcam.py --model-path path/to/model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

from ..common import COCO_CLASSES, resolve_model_path
from ..run_yolo8_onnx import BUNDLED_MODEL_PATH, yolo8_inference_pipeline
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


def run_webcam(pipeline: Pipeline[ImagePayload, ImagePayload]) -> int:
    pipeline.validate()
    pipeline.describe(verbose=True)

    try:
        reader = FrameReader(0)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Running — press Q in the window to quit.", file=sys.stderr)
    while True:
        ok, frame = reader.latest()
        if not ok:
            print("Warning: failed to read frame, stopping.", file=sys.stderr)
            break

        result = pipeline(ImagePayload(array=frame))
        cv2.imshow("YOLOv8 — Webcam", result.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.stop()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live webcam inference with YOLOv8.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to the bundled YOLOv8n model in the assets directory.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)

    pipeline = build_pipeline(model_path)
    return run_webcam(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
