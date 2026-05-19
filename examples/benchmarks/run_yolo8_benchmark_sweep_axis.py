"""
Axis sweep benchmark: sweep slice_wh × overlap_wh axes for the tiled YOLOv8 pipeline.

BenchmarkBuilder expands the cartesian product of all axes automatically.
Useful for finding the tile size / overlap combination that best balances
latency and detection quality.

Axes swept:
  slice_wh   — tile size in pixels (width × height)
  overlap_wh — overlap between adjacent tiles in pixels

A filter drops combinations where overlap >= half the slice size, as those
would produce more than 2× tile coverage and hurt latency with little gain.

Usage:
    python run_yolo8_benchmark_sweep_axis.py
    python run_yolo8_benchmark_sweep_axis.py --runs 20 --warmup 3
    python run_yolo8_benchmark_sweep_axis.py --model s --save results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    __package__ = "examples.benchmarks"

from examples.common import (
    ASSETS_DIR,
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
from examples.run_yolo8_onnx import YOLO8_MODELS
from examples.run_yolo8_tile import yolo8_tiled_pipeline

from ml_pipes import Pipeline, pipeline_factory
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


_DEFAULT_MODEL_VARIANT = "n"
_model_name, _model_url = YOLO8_MODELS[_DEFAULT_MODEL_VARIANT]
_DEFAULT_MODEL_PATH = resolve_model_path(ASSETS_DIR, _model_name, _model_url, _DEFAULT_MODEL_VARIANT)
_DEFAULT_OUTPUT_PATH = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, _model_name)


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path = _DEFAULT_MODEL_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
) -> Pipeline:
    return (
        decode()
        + yolo8_tiled_pipeline(model_path, slice_wh=slice_wh, overlap_wh=overlap_wh)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )


def _input_fn(image_path: Path):
    def fn():
        return image_path.name, image_path, None, None
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per cell (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per cell (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-cell result JSON files.")
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    output_path = build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)

    builder = (
        BenchmarkBuilder.factory(yolo8_tiled_benchmark_pipeline)
        .pipeline_config(model_path=model_path, output_path=output_path)
        .pipeline_config_axis("slice_wh", (240, 240), (320, 320), (480, 480))
        .pipeline_config_axis("overlap_wh", (40, 40), (80, 80), (120, 120))
        .pipeline_config_filter(lambda c: c["overlap_wh"][0] < c["slice_wh"][0] // 2)
        .data_input(_input_fn(image_path))
        .runs(args.runs).warmup(args.warmup)
    )

    results = builder.run(verbose=True)
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
