"""
Tracing example: per-operator latency breakdown for the YOLOv8 pipeline.

Shows how to attach a TraceCollector to an existing pipeline and inspect
the InvocationTrace produced by each call — both a single detailed trace
and a simple aggregate over multiple runs.

Usage:
    python run_yolo8_tracing.py
    python run_yolo8_tracing.py --runs 10
    python run_yolo8_tracing.py --model-path path/to/model.onnx
    python run_yolo8_tracing.py --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    ASSETS_DIR,
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    decode,
    resolve_input_path,
    resolve_model_path,
    visualize_detections_and_store,
)
from run_yolo8_onnx import BUNDLED_MODEL_NAME, yolo8_inference_pipeline
from ml_pipes.collectors import (
    AggregateCollector,
    PrintCollector,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model. Defaults to the bundled yolov8n model in the assets directory.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to a file under the assets directory.")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs for aggregate stats.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, ASSETS_DIR / BUNDLED_MODEL_NAME)
    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, model_path.name)

    infer_pipe = yolo8_inference_pipeline(model_path)
    pipeline = decode() + infer_pipe + visualize_detections_and_store(output_path, COCO_CLASSES)
    pipeline.validate()
    pipeline.describe()

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
