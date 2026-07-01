from __future__ import annotations

import argparse
import collections
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

import cv2
import numpy as np
import supervision as sv

from ..common import COCO_CLASSES, add_assets_dir_arg, add_conf_threshold_arg, add_model_arg, resolve_model_path
from ..run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from .stream_common import FrameReader, add_streaming_args, get_stream_url
from ml_pipes.vision import ImagePayload


def run(url: str, assets_dir: Path, model: str, workers: int, stride: int, tile: bool, conf_threshold: float = 0.25) -> int:
    model_name, model_url = YOLO8_MODELS[model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, model)
    if model_path is None:
        return 1

    pipeline = yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)

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

    try:
        reader = FrameReader(stream_url, stride=stride)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

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
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shibuya crossing stream with ml_pipes ONNX + Supervision.")
    add_streaming_args(parser)
    add_assets_dir_arg(parser)
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
    return run(
        url=args.url,
        assets_dir=args.assets_dir,
        model=args.model,
        workers=args.workers,
        stride=args.stride,
        tile=args.tile,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
