from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from common import COCO_CLASSES, SAMPLE_VIDEO_NAME, SAMPLE_VIDEO_URL, download_if_missing
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

# Sequential frame-by-frame inference on a video file with YOLOv8n.
#
# Reads every frame from a video, runs detection, and writes an annotated
# output video.  This is the single-frame sequential baseline we will later
# compare against batched inference.
#
# Usage:
#   python run_video_yolo8n_onnx.py --input clip.mp4
#   python run_video_yolo8n_onnx.py --input clip.mp4 --output annotated.mp4

MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MODEL_NAME = "yolov8n.onnx"
ASSETS_DIR = Path(".example_assets")


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Store("source_frame"),
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
        Recall("source_frame", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0),
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential YOLOv8n inference on a video file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input video file. Defaults to the bundled sample video (downloaded on first run).",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output annotated video. Defaults to <input>_annotated.mp4.")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.assets_dir / MODEL_NAME

    input_path = args.input or args.assets_dir / SAMPLE_VIDEO_NAME
    if args.input is None:
        print(f"Downloading sample video to {input_path} if needed...", file=sys.stderr)
        download_if_missing(SAMPLE_VIDEO_URL, input_path)

    output_path = args.output or input_path.with_stem(input_path.stem + "_annotated").with_suffix(".mp4")

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

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

    pipeline = build_pipeline(model_path)

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
