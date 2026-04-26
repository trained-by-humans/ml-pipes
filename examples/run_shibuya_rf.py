from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from common import ASSETS_DIR, COCO_CLASSES, add_model_arg, resolve_model_path
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import ImagePayload, Pipeline

# Shibuya crossing live stream using ml_pipes ONNX inference (CoreML-accelerated)
# with Supervision annotators for rendering.
#
# Requires:
#   pip install supervision yt-dlp opencv-python
#
# Usage:
#   python run_shibuya_rf.py
#   python run_shibuya_rf.py --model x --workers 2


def get_stream_url(youtube_url: str) -> str:
    import yt_dlp
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


class FrameReader:
    """Reads frames from a VideoCapture on a background thread into a deque.
    Each frame is tagged with its presentation time (PTS) derived from the
    stream FPS and frame index — independent of HLS burst jitter.

    Two consumption modes:
    - latest(): drops all expired frames, returns the most recent due one (eager).
    - next(): returns the oldest buffered frame in strict arrival order (lazy).

    Both block until a frame is available or the stream ends."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        stream_fps: float,
        stream_url: str,
        reconnect_delay_s: float = 1.0,
        stride: int = 1,
    ) -> None:
        self._cap = cap
        self._stream_url = stream_url
        self._stride = max(1, stride)
        self._frame_interval = 1.0 / stream_fps
        self._reconnect_delay_s = reconnect_delay_s
        self._buf: collections.deque[tuple[float, Any]] = collections.deque()
        self._lock = threading.Lock()
        self._stopped = False
        self._t_start: float | None = None
        self._frame_index = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    break
            ok, frame = self._cap.read()
            if not ok:
                if self._stopped:
                    break
                print("\nStream lost — reconnecting...", file=sys.stderr)
                while not self._stopped:
                    time.sleep(self._reconnect_delay_s)
                    self._cap.open(self._stream_url)
                    if self._cap.isOpened():
                        print("Reconnected.", file=sys.stderr)
                        break
                continue
            now = time.perf_counter()
            if self._t_start is None:
                self._t_start = now
            pts = self._t_start + self._frame_index * self._frame_interval * self._stride
            self._frame_index += 1
            with self._lock:
                self._buf.append((pts, frame))

    def latest(self) -> tuple[bool, Any]:
        while True:
            with self._lock:
                if not self._buf:
                    if self._stopped:
                        return False, None
                else:
                    now = time.perf_counter()
                    current: Any = None
                    while self._buf and self._buf[0][0] <= now:
                        _, current = self._buf.popleft()
                    if current is not None:
                        return True, current
            time.sleep(0.001)

    def stop(self) -> None:
        self._stopped = True
        self._thread.join()


def build_pipeline(model_path: Path) -> Pipeline:
    return yolo8_inference_pipeline(model_path)


def run(url: str, assets_dir: Path, model: str, workers: int, stride: int, tile: bool) -> int:
    model_name, model_url = YOLO8_MODELS[model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, model)
    if model_path is None:
        return 1

    pipeline = build_pipeline(model_path)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    fps_monitor = sv.FPSMonitor()

    def _detect(frame: np.ndarray) -> sv.Detections:
        source = ImagePayload(array=frame, color_space="BGR", layout="HWC")
        result = pipeline(source)
        return sv.Detections(
            xyxy=np.array(result.boxes, dtype=np.float32),
            confidence=np.array(result.scores, dtype=np.float32),
            class_id=np.array(result.classes, dtype=int),
        )

    slicer = sv.InferenceSlicer(
        callback=_detect,
        slice_wh=(640, 640),
        overlap_wh=(100, 100),
        overlap_filter=sv.OverlapFilter.NON_MAX_MERGE,
        iou_threshold=0.5,
        thread_workers=1,
    ) if tile else None

    print(f"Resolving stream URL from {url} ...", file=sys.stderr)
    stream_url = get_stream_url(url)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: could not open stream.", file=sys.stderr)
        return 1

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    reader = FrameReader(cap, stream_fps=stream_fps, stream_url=stream_url, stride=stride)
    mode = "tiled" if tile else "standard"
    print(f"Streaming with {workers} worker(s), stride={stride}, mode={mode} — press Q in the window to quit.", file=sys.stderr)

    def infer(frame: Any) -> Any:
        detections = slicer(frame) if tile else _detect(frame)
        labels = [
            f"{COCO_CLASSES[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = box_annotator.annotate(frame.copy(), detections)
        return label_annotator.annotate(annotated, detections, labels=labels)

    pending: collections.deque[Future] = collections.deque()
    stopped = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            while not stopped and len(pending) < workers:
                ok, frame = reader.latest()
                if not ok or frame is None:
                    stopped = True
                    break
                pending.append(pool.submit(infer, frame))

            if not pending:
                break

            annotated = pending.popleft().result()
            fps_monitor.tick()
            cv2.putText(
                annotated,
                f"FPS: {fps_monitor.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Shibuya Crossing", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped = True

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shibuya crossing stream with ml_pipes ONNX + Supervision.")
    parser.add_argument("--url", default="https://www.youtube.com/watch?v=dfVK7ld38Ys")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    add_model_arg(parser, list(YOLO8_MODELS), default="x")
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable SAHI-style tiling for better small object detection (slower).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        url=args.url,
        assets_dir=args.assets_dir,
        model=args.model,
        workers=args.workers,
        stride=args.stride,
        tile=args.tile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
