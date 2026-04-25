from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from common import COCO_CLASSES, download_if_missing
from run_yolo8n_onnx import MODEL_NAME, MODEL_URL, yolo8n_inference_pipeline
from ml_pipes import (
    DrawBoxes,
    Embed,
    ImagePayload,
    Pick,
    Pipeline,
    Recall,
    Store,
    ThroughputCollector,
)


def get_stream_url(youtube_url: str) -> str:
    import yt_dlp
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Store("source_frame"),
        Embed(yolo8n_inference_pipeline(model_path)),
        Recall("source_frame", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0),
    ])


def run_pipeline(url: str, assets_dir: Path, target_fps: float) -> int:
    model_path = assets_dir / MODEL_NAME
    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    throughput = ThroughputCollector(target_fps=target_fps)
    pipeline = build_pipeline(model_path)
    pipeline.set_tracing(throughput)

    print(f"Resolving stream URL from {url} ...", file=sys.stderr)
    stream_url = get_stream_url(url)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: could not open stream.", file=sys.stderr)
        return 1

    print("Streaming — press Q in the window to quit.", file=sys.stderr)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Warning: failed to read frame, stopping.", file=sys.stderr)
            break

        source = ImagePayload(array=frame, color_space="BGR", layout="HWC")
        annotated = pipeline(source)

        cv2.imshow("Shibuya Crossing — YOLOv8n", annotated.array)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    throughput.flush()
    throughput.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Shibuya crossing with live YOLOv8n detections.")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=dfVK7ld38Ys",
        help="YouTube live URL.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(".example_assets"),
        help="Directory used to cache the downloaded model.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Target FPS for throughput reporting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(url=args.url, assets_dir=args.assets_dir, target_fps=args.target_fps)


if __name__ == "__main__":
    raise SystemExit(main())
