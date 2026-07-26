"""
Lowest-level benchmarking example for YOLOv8.

This script uses `Benchmark` directly rather than `BenchmarkBuilder` or the
CLI. It is the best reference when you want to inspect a single measured run
and compare the resulting `BenchmarkResult` objects with `diff()`.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark.py --model n --runs 30
    python examples/benchmarks/run_yolo8_benchmark.py --model s --save results/

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
    visualize_detections_and_store,
)
from examples.benchmarks.benchmark_common import YOLO8_MODELS, resolve_model_variant_path
from examples.run_yolo8_onnx import yolo8_inference_pipeline

from ml_pipes.core import Pipeline
from ml_pipes.benchmark import Benchmark, BenchmarkResult, MeasurementConfig


def _input_fn(image_path: Path):
    """Returns a fixed input driver: always feeds the same image."""
    def fn():
        return (image_path.name, image_path, None, None)
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(YOLO8_MODELS),
        default="n",
        help=f"Model variant ({' → '.join(YOLO8_MODELS)}).",
    )
    parser.add_argument("--runs", type=int, default=30, help="Measured runs per benchmark (default: 30).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs discarded before measurement (default: 5).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save result JSON files.")
    args = parser.parse_args()

    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_variant_path(model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)

    config = MeasurementConfig(runs=args.runs, warmup=args.warmup, percentiles=(0.50, 0.95, 0.99))
    input_fn = _input_fn(image_path)

    # ------------------------------------------------------------------
    # 1. Single pipeline — full breakdown
    # ------------------------------------------------------------------
    print("\n=== 1. Single pipeline benchmark ===\n")
    output_path = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, model_name)
    full_pipeline = (
        decode()
        + yolo8_inference_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )

    result_full = Benchmark(
        pipeline=full_pipeline,
        input_fn=input_fn,
        measurement=config,
        label=f"yolo8{args.model}-full",
        metadata={"model": model_name, "variant": args.model},
    ).run()

    print(result_full.to_table())

    # ------------------------------------------------------------------
    # 2. Config comparison — low vs high confidence threshold
    # ------------------------------------------------------------------
    print("\n=== 2. Config comparison: conf_threshold=0.10 vs 0.50 ===\n")
    pipeline_low_conf = (
        decode()
        + yolo8_inference_pipeline(model_path, conf_threshold=0.10)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline_high_conf = (
        decode()
        + yolo8_inference_pipeline(model_path, conf_threshold=0.50)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )

    result_low = Benchmark(
        pipeline=pipeline_low_conf,
        input_fn=input_fn,
        measurement=config,
        label="conf=0.10",
        metadata={"conf_threshold": 0.10},
    ).run()

    result_high = Benchmark(
        pipeline=pipeline_high_conf,
        input_fn=input_fn,
        measurement=config,
        label="conf=0.50",
        metadata={"conf_threshold": 0.50},
    ).run()

    diff_conf = result_low.diff(result_high)
    print(diff_conf.to_table())

    # ------------------------------------------------------------------
    # 3. Structural comparison — inference-only vs full pipeline
    # ------------------------------------------------------------------
    print("\n=== 3. Structural comparison: inference-only vs full pipeline ===\n")
    infer_only_pipeline = decode() + yolo8_inference_pipeline(model_path)

    result_infer_only = Benchmark(
        pipeline=infer_only_pipeline,
        input_fn=input_fn,
        measurement=config,
        label="inference-only",
        metadata={"model": model_name},
    ).run()

    diff_structural = result_infer_only.diff(result_full)
    print(diff_structural.to_table())
    print("\n(operators with 'only in candidate' are the visualisation steps added in the full pipeline)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save:
        save_dir = args.save
        save_dir.mkdir(parents=True, exist_ok=True)
        result_full.save(str(save_dir / f"yolo8{args.model}_full.json"))
        result_infer_only.save(str(save_dir / f"yolo8{args.model}_infer_only.json"))
        result_low.save(str(save_dir / f"yolo8{args.model}_conf_low.json"))
        result_high.save(str(save_dir / f"yolo8{args.model}_conf_high.json"))
        print(f"\nResults saved to {save_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
