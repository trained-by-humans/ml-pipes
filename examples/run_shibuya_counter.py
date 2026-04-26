from __future__ import annotations

import argparse
import sys
import collections
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from common import COCO_CLASSES, add_assets_dir_arg, add_conf_threshold_arg, add_model_arg, resolve_model_path
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import (
    DrawBoxes,
    Embed,
    ImagePayload,
    Pick,
    Pipeline,
    Recall,
    Store,
    ThroughputCollector,
    Detections,
)


def get_stream_url(youtube_url: str) -> str:
    import yt_dlp
    with yt_dlp.YoutubeDL({"format": "best[ext=mp4]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


class InferenceSlicer:
    """Wraps sv.InferenceSlicer to present the same ImagePayload → Detections
    contract as a plain inference pipeline."""

    def __init__(self, base: Pipeline) -> None:
        def _tile_callback(tile_arr: np.ndarray) -> sv.Detections:
            result: Detections = base(ImagePayload(array=tile_arr, color_space="BGR", layout="HWC"))
            return sv.Detections(
                xyxy=np.array(result.boxes, dtype=np.float32),
                confidence=np.array(result.scores, dtype=np.float32),
                class_id=np.array(result.classes, dtype=int),
            )

        self.slicer = sv.InferenceSlicer(
            callback=_tile_callback,
            slice_wh=(640, 640),
            overlap_wh=(100, 100),
            overlap_filter=sv.OverlapFilter.NON_MAX_MERGE,
            iou_threshold=0.5,
            thread_workers=1,
        )

    def __call__(self, payload: ImagePayload) -> Detections:
        sv_dets = self.slicer(payload.array)
        return Detections(
            boxes=[box.tolist() for box in sv_dets.xyxy],
            scores=sv_dets.confidence.tolist(),
            classes=sv_dets.class_id.tolist(),
        )


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
            # Grab and discard stride-1 frames, then decode the strided one
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
            # PTS advances by stride intervals so timing stays true to stream clock
            pts = self._t_start + self._frame_index * self._frame_interval * self._stride
            self._frame_index += 1
            with self._lock:
                self._buf.append((pts, frame))

    def latest(self) -> tuple[bool, Any]:
        """Eager: drop expired frames, return the most recent due frame."""
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

    def next(self) -> tuple[bool, Any]:
        """Lazy: return the oldest buffered frame in strict arrival order."""
        while True:
            with self._lock:
                if self._buf:
                    _, frame = self._buf.popleft()
                    return True, frame
                if self._stopped:
                    return False, None
            time.sleep(0.001)

    def stop(self) -> None:
        self._stopped = True
        self._thread.join()


def build_pipeline(model_path: Path, conf_threshold: float, tile: bool) -> Pipeline:
    infer_pipe = yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)
    if tile:
        infer_pipe = Pipeline([InferenceSlicer(infer_pipe)])
    return Pipeline([
        Store("source_frame"),
        Embed(infer_pipe),
        Recall("source_frame", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0),
    ])


def run_pipeline(url: str, assets_dir: Path, target_fps: float, workers: int, stride: int, model: str, tile: bool, conf_threshold: float) -> int:
    model_name, model_url = YOLO8_MODELS[model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, model)
    if model_path is None:
        return 1

    throughput = ThroughputCollector(target_fps=target_fps, report_interval_s=1.0)
    pipeline = build_pipeline(model_path, conf_threshold, tile)
    pipeline.set_tracing(throughput)

    print(f"Resolving stream URL from {url} ...", file=sys.stderr)
    stream_url = get_stream_url(url)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: could not open stream.", file=sys.stderr)
        return 1

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    throughput.target_fps = stream_fps
    reader = FrameReader(cap, stream_fps=stream_fps, stream_url=stream_url, stride=stride)
    mode = "tiled" if tile else "standard"
    print(f"Streaming with {workers} worker(s), stride={stride}, mode={mode} — press Q in the window to quit.", file=sys.stderr)

    pending: collections.deque[Future] = collections.deque()
    stopped = False

    def infer(frame: Any) -> Any:
        return pipeline(ImagePayload(array=frame, color_space="BGR", layout="HWC"))

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

            result = pending.popleft().result()
            cv2.imshow("Shibuya Crossing - YOLOv8", result.array)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped = True

    reader.stop()
    cap.release()
    cv2.destroyAllWindows()
    throughput.flush()
    throughput.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Shibuya crossing with live YOLOv8 detections.")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/watch?v=dfVK7ld38Ys",
        help="YouTube live URL.",
    )
    add_assets_dir_arg(parser)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Target FPS for throughput reporting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel inference workers.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame; intermediate frames are grabbed but not decoded.",
    )
    add_model_arg(parser, list(YOLO8_MODELS), default="x")
    add_conf_threshold_arg(parser)
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable SAHI-style tiling for better small object detection (slower).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(
        url=args.url,
        assets_dir=args.assets_dir,
        target_fps=args.target_fps,
        workers=args.workers,
        stride=args.stride,
        model=args.model,
        tile=args.tile,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
