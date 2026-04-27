from __future__ import annotations

import argparse
import collections
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

import cv2
import numpy as np
import supervision as sv

from ..common import COCO_CLASSES, add_assets_dir_arg, add_conf_threshold_arg, add_model_arg, resolve_model_path
from ..run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from .stream_common import FrameReader, add_streaming_args, get_stream_url
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

    try:
        reader = FrameReader(stream_url, fallback_fps=target_fps, stride=stride)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    throughput.target_fps = reader.stream_fps
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
    cv2.destroyAllWindows()
    throughput.flush()
    throughput.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Shibuya crossing with live YOLOv8 detections.")
    add_streaming_args(parser)
    add_assets_dir_arg(parser)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Target FPS for throughput reporting.",
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
