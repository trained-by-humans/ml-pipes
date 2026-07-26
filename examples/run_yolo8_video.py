from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from common import ASSETS_DIR, COCO_CLASSES, SAMPLE_VIDEO_NAME, SAMPLE_VIDEO_URL, resolve_input_path, resolve_model_path
from run_yolo8_onnx import BUNDLED_MODEL_NAME, yolo8_inference_pipeline
from ml_pipes.core import (
    Pipeline,
    Embed,
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

# Sequential frame-by-frame inference on a video file with YOLOv8.
#
# Reads every frame from a video, runs detection, and writes an annotated
# output video.  This is the single-frame sequential baseline we will later
# compare against batched inference.
#
# Usage (shown from `examples/`; from repo root, prefix script paths with
# `examples/`):
#   python run_yolo8_video.py --input clip.mp4
#   python run_yolo8_video.py --input clip.mp4 --output annotated.mp4

def build_video_annotation_pipeline(model_path: Path) -> Pipeline[ImagePayload, ImagePayload]:
    return Pipeline([
        Store("source_frame"),
        Embed(yolo8_inference_pipeline(model_path)),
        Recall("source_frame", prepend=True),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0)
    ], auto_validate=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential YOLOv8 inference on a video file.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory used to cache downloaded models and sample assets.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model. Defaults to the bundled yolov8n model in the assets directory.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input video path. Defaults to the bundled sample video.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output video path. Defaults to <input>_annotated.mp4.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, args.assets_dir, BUNDLED_MODEL_NAME)
    input_path = resolve_input_path(args.input, args.assets_dir, SAMPLE_VIDEO_NAME, SAMPLE_VIDEO_URL)

    output_path = args.output or input_path.with_stem(input_path.stem + "_annotated").with_suffix(".mp4")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {input_path}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    pipeline = build_video_annotation_pipeline(model_path)
    pipeline.describe()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        source = ImagePayload(array=frame, color_space="BGR", layout="HWC")
        annotated = pipeline(source)
        writer.write(annotated.array)

        frame_idx += 1
        print(f"\r  {frame_idx}/{total} frames", end="", file=sys.stderr)

    print(file=sys.stderr)
    cap.release()
    writer.release()
    print(f"Saved annotated video to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
