"""
Tracing example: per-operator latency breakdown for the YOLOv8n pipeline.

Shows how to attach a TraceCollector to an existing pipeline and inspect
the InvocationTrace produced by each call — both a single detailed trace
and a simple aggregate over multiple runs.

Usage:
    python run_yolo8n_tracing.py
    python run_yolo8n_tracing.py --runs 10 --assets-dir /tmp/assets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    download_if_missing,
    visualize_detections_and_store,
)
from ml_pipes import (
    AggregateCollector,
    ArgMax,
    ConvertBoxFormat,
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
)

MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MODEL_NAME = "yolov8n.onnx"


def yolo8n_inference_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
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
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assets-dir", type=Path, default=Path(".example_assets"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs for aggregate stats.")
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, MODEL_NAME)

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)
    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    infer_pipe = yolo8n_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)

    # --- single detailed trace (warm-up run) ---
    print("\n=== Single invocation trace (warm-up run) ===\n")
    pipeline.set_tracing(PrintCollector())
    pipeline(image_path)
    pipeline.set_tracing(None)

    # --- aggregate over N runs ---
    print(f"\n=== Aggregate over {args.runs} runs ===\n")
    agg = AggregateCollector()
    pipeline.set_tracing(agg)
    for _ in range(args.runs):
        pipeline(image_path)
    pipeline.set_tracing(None)
    agg.print_summary()

    print(f"\nAnnotated image saved to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
