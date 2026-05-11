"""
Sweep benchmark example: sweep conf_threshold configs × image inputs for YOLOv8.

Runs every (pipeline config, input) combination and renders a multi-column
comparison table — mean and percentiles per operator for each combination.

Each row is an operator; each group of columns is one (input, config) pair.
This makes it easy to spot which operator cost changes as confidence threshold
rises, and whether the effect is input-dependent.

Demonstrates BenchmarkSweep as the explicit-list layer on top of Benchmark:

  - pipeline_factory  builds a fresh pipeline for each config dict
  - pipeline_configs  list of dicts to sweep (here: conf_threshold values)
  - inputs            list of InputFn callables (here: the same image repeated,
                      but the pattern extends to multiple distinct images/datasets)

Results are also available as list[BenchmarkResult] for downstream use with
MLflow, W&B, or any JSON-compatible store via result.to_dict().

Usage:
    python run_yolo8_matrix_benchmark.py
    python run_yolo8_matrix_benchmark.py --runs 20 --warmup 3
    python run_yolo8_matrix_benchmark.py --model s --save results/
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
from examples.run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline

from ml_pipes import Pipeline
from ml_pipes.benchmark import BenchmarkSweep, MeasurementConfig


def _make_pipeline(model_path: Path, output_path: Path, coco_classes: list[str]):
    """Return a pipeline factory for the given model and output path."""
    def factory(config: dict) -> Pipeline:
        conf = config.get("conf_threshold", 0.25)
        return (
            decode()
            + yolo8_inference_pipeline(model_path, conf_threshold=conf)
            + visualize_detections_and_store(output_path, coco_classes)
        )
    return factory


def _input_fn(image_path: Path, label: str | None = None):
    """Return an InputFn that always feeds the same image."""
    name = label or image_path.name
    def fn():
        return (name, image_path, None, None)
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

    pipeline_configs = [
        {"conf_threshold": 0.10},
        {"conf_threshold": 0.25},
        {"conf_threshold": 0.50},
        {"conf_threshold": 0.75},
    ]

    # Two inputs: the same image labelled differently to show the pattern.
    # In practice, swap these for distinct images or dataset splits.
    inputs = [
        _input_fn(image_path, label="coco_sample"),
    ]

    config = MeasurementConfig(runs=args.runs, warmup=args.warmup, percentiles=(0.50, 0.95, 0.99))

    print(f"\nRunning {len(inputs)} input(s) × {len(pipeline_configs)} config(s) "
          f"= {len(inputs) * len(pipeline_configs)} cells "
          f"({args.warmup} warmup + {args.runs} measured each)\n", file=sys.stderr)

    matrix = BenchmarkSweep(
        pipeline_factory=_make_pipeline(model_path, output_path, COCO_CLASSES),
        pipeline_configs=pipeline_configs,
        inputs=inputs,
        config=config,
    )

    results = matrix.run()
    print(BenchmarkSweep.to_table(results))

    if args.save:
        save_dir = args.save
        save_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.save(str(save_dir / result.slug(".json")))
        print(f"\nResults saved to {save_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
