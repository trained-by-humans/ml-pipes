"""
Tracing example: per-operator latency breakdown for the YOLOv8 pipeline.

Shows how to attach a TraceCollector to an existing pipeline and inspect
the InvocationTrace produced by each call — both a single detailed trace
and a simple aggregate over multiple runs.

Usage:
    python run_yolo8_tracing.py
    python run_yolo8_tracing.py --runs 10 --assets-dir /tmp/assets
    python run_yolo8_tracing.py --model s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_model_arg,
    build_output_path,
    decode,
    download_if_missing,
    resolve_model_path,
    visualize_detections_and_store,
)
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import (
    AggregateCollector,
    Pipeline,
    PrintCollector,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs for aggregate stats.")
    add_model_arg(parser, list(YOLO8_MODELS))
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    infer_pipe = yolo8_inference_pipeline(model_path)
    pipeline: Pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline.validate()

    # --- single detailed trace (warm-up run) ---
    print("\n=== Single invocation trace (warm-up run) ===\n")
    pipeline.set_tracing(PrintCollector(), capture_shapes=True)
    pipeline(image_path)
    pipeline.set_tracing(None, capture_shapes=False)

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
