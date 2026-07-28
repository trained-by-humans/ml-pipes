"""
Lowest-level benchmarking example for YOLOv8.

This script uses `Benchmark` directly rather than `BenchmarkBuilder` or the
CLI. It is the best reference when you want to inspect one measured run,
then compare it directly against a second result with `diff()` before moving
on to the higher-level sweep or CLI examples.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark.py --runs 30
    python examples/benchmarks/run_yolo8_benchmark.py --model-path path/to/model.onnx --save results/

See `docs/BENCHMARKING.md` for shared measurement options, result formats, and the
higher-level sweep / CLI APIs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.benchmarks"

from examples.common import (
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
from examples.run_yolo8_onnx import BUNDLED_MODEL_PATH, yolo8_inference_pipeline

from ml_pipes.benchmark import Benchmark, MeasurementConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model. Defaults to the bundled yolov8n model.",
    )
    parser.add_argument("--runs", type=int, default=30, help="Measured runs per benchmark (default: 30).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs discarded before measurement (default: 5).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save result JSON files.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    output_path = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, model_path.name)

    config = MeasurementConfig(runs=args.runs, warmup=args.warmup, percentiles=(0.50, 0.95, 0.99))
    input_fn = lambda: (image_path.name, image_path, None, None)

    full_pipeline = (
        decode()
        + yolo8_inference_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )

    result_full = Benchmark(
        pipeline=full_pipeline,
        input_fn=input_fn,
        measurement=config,
        label=f"{model_path.stem}-full",
        metadata={"model": model_path.name},
    ).run()

    print(result_full.to_table())

    print("\n=== Structural comparison: inference-only vs full pipeline ===\n")
    infer_only_pipeline = decode() + yolo8_inference_pipeline(model_path)

    result_infer_only = Benchmark(
        pipeline=infer_only_pipeline,
        input_fn=input_fn,
        measurement=config,
        label=f"{model_path.stem}-infer-only",
        metadata={"model": model_path.name},
    ).run()

    diff = result_infer_only.diff(result_full)
    print(diff.to_table())
    print("\n(operators with 'only in candidate' are the visualisation steps added in the full pipeline)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save:
        save_dir = args.save
        save_dir.mkdir(parents=True, exist_ok=True)
        result_full.save(str(save_dir / result_full.slug(".json")))
        result_infer_only.save(str(save_dir / result_infer_only.slug(".json")))
        print(f"\nResults saved to {save_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
