from __future__ import annotations

import argparse
import collections
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.streaming"

import cv2
from ..common import COCO_CLASSES, resolve_model_path
from ..run_yolo8_onnx import BUNDLED_MODEL_PATH
from .stream_common import FrameReader, add_streaming_args, resolve_stream_source
from ml_pipes.collectors import ThroughputCollector
from ml_pipes.core import (
    Pipeline,
    Inline,
)
from ml_pipes.onnx import (
    Extract,
    Infer,
)
from ml_pipes.standard import (
    Gather,
    Pick,
    Recall,
    Scatter,
    Store,
)
from ml_pipes.tensor import (
    ArgMax,
    GatherScores,
    Slice,
    Squeeze,
    Transpose,
)
from ml_pipes.vision import (
    ConvertBoxFormat,
    Detections,
    DrawBoxes,
    FilterPredictionsByArea,
    FilterPredictionsByClass,
    FilterTensorsByClasses,
    ImagePayload,
    NMS,
    NMM,
    Normalize,
    ProjectBoxes,
    Resize,
    Stitch,
    Tile,
    ToDetections,
)

_KEEP_CLASSES = {0, 2, 25}  # COCO: 0=person, 2=car, 25=umbrella
_MAX_HUMAN_AREA = 1_000  # px² — filters out cars and other large objects


def _infer_pipeline(model_path: Path, conf_threshold: float) -> Pipeline[ImagePayload, Detections]:
    return Pipeline([
        Resize((640, 640)),
        Store("resize_transform", source=1),
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
        FilterTensorsByClasses(
            "boxes",
            "scores",
            "classes",
            keep_classes=_KEEP_CLASSES,
        ),
        ConvertBoxFormat(from_="cxcywh"),
        NMS(conf_threshold=conf_threshold),
        Recall("resize_transform"),
        ProjectBoxes(),
        ToDetections(),
    ])


def build_pipeline(
    model_path: Path,
    conf_threshold: float,
    tile: bool,
) -> Pipeline[ImagePayload, ImagePayload]:
    preprocess = Pipeline([
        Store("source_frame"),
    ])

    core_pipeline = Pipeline([
        Tile(slice_wh=(240, 240), overlap_wh=(80, 80)),
        Store("tile_rects", source=1),
        Pick(0),
        Scatter(max_concurrency=6),
        Inline(_infer_pipeline(model_path, conf_threshold)),
        Gather(),
        Recall("tile_rects"),
        Stitch(),
        NMM(iou_threshold=0.5),
    ], auto_validate=True) if tile else _infer_pipeline(model_path, conf_threshold)

    postprocess = Pipeline([
        FilterPredictionsByClass(_KEEP_CLASSES),
        FilterPredictionsByArea(max_area=_MAX_HUMAN_AREA),
        Recall("source_frame", prepend=True),
        DrawBoxes(class_names=COCO_CLASSES),
        Pick(0),
    ])

    return preprocess + core_pipeline + postprocess


def run_stream(
    url: str,
    pipeline: Pipeline[ImagePayload, ImagePayload],
    target_fps: float,
    workers: int,
    stride: int,
    mode: str,
) -> int:
    print(f"Resolving stream source from {url} ...", file=sys.stderr)

    try:
        stream_source = resolve_stream_source(url)
        reader = FrameReader(stream_source, fallback_fps=target_fps, stride=stride)
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    throughput = ThroughputCollector(target_fps=target_fps, report_interval_s=1.0)
    pipeline.set_tracing(throughput)
    throughput.target_fps = reader.stream_fps
    pending: collections.deque[Future[ImagePayload]] = collections.deque()
    stopped = False
    status = f"mode={mode}"
    window_title = "Shibuya Crossing - YOLOv8"

    print(
        f"Streaming with {workers} worker(s), stride={stride}, {status} "
        "— press Q in the window to quit.",
        file=sys.stderr,
    )

    def infer(frame: np.ndarray) -> ImagePayload:
        return pipeline(ImagePayload(array=frame))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped = True

            while not stopped and len(pending) < workers:
                ok, frame = reader.latest()
                if not ok or frame is None:
                    stopped = True
                    break
                pending.append(pool.submit(infer, frame))

            if not pending:
                break

            future = pending[0]
            try:
                result = future.result(timeout=0.05)
                pending.popleft()
                cv2.imshow(window_title, result.array)
            except TimeoutError:
                pass

    reader.stop()
    cv2.destroyAllWindows()
    throughput.flush()
    throughput.print_summary()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Shibuya crossing with live YOLOv8 detections.")
    add_streaming_args(parser)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Fallback FPS when the stream does not expose one.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model. Defaults to the bundled yolov8n model in the assets directory.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence score for detections (default: 0.25).",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable SAHI-style tiling for better small object detection (slower).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    pipeline = build_pipeline(model_path, args.conf_threshold, args.tile)
    pipeline.validate()
    mode = "tiled" if args.tile else "standard"
    return run_stream(
        url=args.url,
        pipeline=pipeline,
        target_fps=args.target_fps,
        workers=args.workers,
        stride=args.stride,
        mode=mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
