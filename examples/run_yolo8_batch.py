from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import add_assets_dir_arg, COCO_IMAGE_NAME, COCO_IMAGE_URL, download_if_missing
from ml_pipes import (
    ArgMax,
    Batch,
    Collate,
    ConvertBoxFormat,
    Decode,
    Detections,
    Distribute,
    LoadFile,
    GatherScores,
    Infer,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    PrintCollector,
    ProjectBoxes,
    Recall,
    Resize,
    Extract,
    Slice,
    Squeeze,
    Store,
    ToDetections,
    Transpose,
    UnBatch,
)

# Batched YOLOv8 detection across multiple images using concurrent threads.
#
# Batch and UnBatch delimit a batch region in the pipeline.  When enough
# threads have reached Batch (or the timeout fires), one thread becomes the
# leader and runs Collate → Infer → Distribute as a single batched call.
# The other threads wait and resume with their individual result after UnBatch.
#
# The model is exported from Ultralytics with dynamic=True so ONNX Runtime
# accepts any batch size.  Requires: pip install ultralytics
#
# Usage (runs the sample COCO image 8 times by default):
#   python run_yolo8_batch.py
#
# Run with explicit images:
#   python run_yolo8_batch.py --images img1.jpg img2.jpg img3.jpg img4.jpg
#
# Tune batch size and thread-pool size:
#   python run_yolo8_batch.py --batch-size 4 --workers 8

MODEL_NAME = "yolov8n_dynamic.onnx"


def _export_dynamic_model(dst: Path) -> None:
    """Export YOLOv8n (nano) with a dynamic batch axis using Ultralytics."""
    from ultralytics import YOLO

    dst.parent.mkdir(parents=True, exist_ok=True)
    pt_path = dst.parent / "yolov8n.pt"
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", dynamic=True, imgsz=640, simplify=False)
    Path(exported).rename(dst)
    pt_path.unlink(missing_ok=True)


def build_pipeline(model_path: Path, batch_size: int, timeout: float,
                   serialize: bool = False) -> Pipeline[str | Path, Detections]:
    return Pipeline([
        LoadFile(),
        Decode(),
        Resize((640, 640)),
        Store("resize_transform", source=1),
        Pick(0),
        Normalize(),
        Batch(size=batch_size, timeout=timeout),
        Collate(),
        Infer(model_path, serialize=serialize, providers=["CPUExecutionProvider"]),
        Distribute(),
        UnBatch(),
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
    ], auto_validate=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched YOLOv8 detection across multiple images.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more image paths. Defaults to the sample COCO image "
            "repeated --workers times."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Max number of images per inference batch (default: 4).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread-pool size — set equal to or a multiple of --batch-size (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.05,
        help="Seconds to wait before running a partial batch (default: 0.05).",
    )
    add_assets_dir_arg(parser)
    parser.add_argument(
        "--no-serialize",
        action="store_true",
        help="Disable the inference lock (allows concurrent session.run() calls).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME

    if not model_path.exists():
        print(f"Exporting YOLOv8n nano (dynamic batch) → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

    if args.images is not None:
        image_paths = args.images
    else:
        sample = assets_dir / COCO_IMAGE_NAME
        print(f"Downloading sample image to {sample} if needed...", file=sys.stderr)
        download_if_missing(COCO_IMAGE_URL, sample)
        # Repeat the sample image to fill the thread pool and demonstrate batching.
        image_paths = [sample] * args.workers

    pipeline = build_pipeline(model_path, args.batch_size, args.timeout,
                              serialize=not args.no_serialize)
    pipeline.describe()
    pipeline.set_tracing(PrintCollector())

    print(
        f"\nRunning {len(image_paths)} images"
        f" | batch_size={args.batch_size}"
        f" | workers={args.workers}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pipeline, p): p for p in image_paths}
        for future in as_completed(futures):
            path = futures[future]
            detections = future.result()
            print(f"  {path.name}: {len(detections.boxes)} detections")
    elapsed = time.perf_counter() - t0

    print(
        f"\nProcessed {len(image_paths)} images in {elapsed:.3f}s"
        f" ({elapsed / len(image_paths) * 1000:.1f} ms/image)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
