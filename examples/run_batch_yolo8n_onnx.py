from __future__ import annotations

# Batched YOLOv8n detection across multiple images using concurrent threads.
#
# Batch and UnBatch delimit a batch region in the pipeline.  When enough
# threads have reached Batch (or the timeout fires), one thread becomes the
# leader and runs Collate → Infer → Distribute as a single batched call.
# The other threads wait and resume with their individual result after UnBatch.
#
# Usage (downloads the sample COCO image and runs it 8 times by default):
#   python run_batch_yolo8n_onnx.py
#
# Run with explicit images:
#   python run_batch_yolo8n_onnx.py --images img1.jpg img2.jpg img3.jpg img4.jpg
#
# Tune batch size and thread-pool size:
#   python run_batch_yolo8n_onnx.py --batch-size 4 --workers 8

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ml_pipes import (
    ArgMax,
    Batch,
    Collate,
    ConvertBoxFormat,
    Decode,
    Distribute,
    GatherScores,
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
    UnBatch,
)
from common import COCO_IMAGE_NAME, COCO_IMAGE_URL, download_if_missing


MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MODEL_NAME = "yolov8n.onnx"
ASSETS_DIR = Path(".example_assets")


def build_pipeline(model_path: Path, batch_size: int, timeout: float) -> Pipeline:
    batch = Batch(size=batch_size, timeout=timeout)
    unbatch = UnBatch()

    return Pipeline([
        Decode(),
        Resize((640, 640)),
        Store("resize_transform", index=1),
        Pick(0),
        Normalize(),
        batch,
        Collate(),
        Infer(model_path),
        Distribute(),
        unbatch,
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
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched YOLOv8n detection across multiple images.",
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
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    if args.images is not None:
        image_paths = args.images
    else:
        sample = assets_dir / COCO_IMAGE_NAME
        print(f"Downloading sample image to {sample} if needed...", file=sys.stderr)
        download_if_missing(COCO_IMAGE_URL, sample)
        # Repeat the sample image to fill the thread pool and demonstrate batching.
        image_paths = [sample] * args.workers

    pipeline = build_pipeline(model_path, args.batch_size, args.timeout)

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
