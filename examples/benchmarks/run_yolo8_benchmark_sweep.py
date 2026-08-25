"""
`BenchmarkBuilder` sweep example for plain vs tiled YOLOv8 inference.

This script keeps the sweep intentionally small:
- one plain baseline result
- one tiled parameter sweep over `slice_wh`

Use it when you want one comparison table without jumping between separate
sweep scripts.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark_sweep.py --runs 20
    python examples/benchmarks/run_yolo8_benchmark_sweep.py --model-path path/to/model.onnx --save results/

See `docs/BENCHMARKING.md` for shared sweep concepts, measurement options, and
factory rules.
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
from examples.run_yolo8_tile import yolo8_tiled_pipeline

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload


@pipeline_factory
def yolo8_plain_benchmark_pipeline(
    model_path: Path | None = None,
    output_path: Path | None = None,
) -> Pipeline[str | Path, tuple[ImagePayload, TensorRegistry]]:
    resolved_model_path = resolve_model_path(model_path, BUNDLED_MODEL_PATH)
    resolved_output_path = output_path or build_output_path(ASSETS_DIR, "run_yolo8_benchmark_sweep_plain.jpg", resolved_model_path.name)
    return (
        decode()
        + yolo8_inference_pipeline(resolved_model_path)
        + visualize_detections_and_store(resolved_output_path, COCO_CLASSES)
    )


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path | None = None,
    output_path: Path | None = None,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
) -> Pipeline[str | Path, tuple[ImagePayload, TensorRegistry]]:
    resolved_model_path = resolve_model_path(model_path, BUNDLED_MODEL_PATH)
    resolved_output_path = output_path or build_output_path(ASSETS_DIR, "run_yolo8_benchmark_sweep_tiled.jpg", resolved_model_path.name)
    return (
        decode()
        + yolo8_tiled_pipeline(
            resolved_model_path,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
        )
        + visualize_detections_and_store(resolved_output_path, COCO_CLASSES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to the bundled YOLOv8n model in the assets directory.")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per cell (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per cell (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-cell result JSON files.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)

    input_fn = lambda: (image_path.name, image_path, None, None)

    plain_results = (
        BenchmarkBuilder.factory(yolo8_plain_benchmark_pipeline)
        .pipeline_config(model_path=model_path)
        .data_input(input_fn)
        .runs(args.runs).warmup(args.warmup)
        .run(verbose=False)
    )

    tiled_results = (
        BenchmarkBuilder.factory(yolo8_tiled_benchmark_pipeline)
        .pipeline_config(model_path=model_path)
        .pipeline_config_axis("slice_wh", (320, 320), (480, 480))
        .data_input(input_fn)
        .runs(args.runs).warmup(args.warmup)
        .run(verbose=False)
    )

    results = plain_results + tiled_results
    print(BenchmarkResult.to_comparison_table(results, expand_regions=False))

    if args.save:
        save_dir = args.save
        save_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.save(str(save_dir / result.slug(".json")))
        print(f"\nResults saved to {save_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
