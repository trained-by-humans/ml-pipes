"""
Sweep benchmark: plain inference vs tiled inference for YOLOv8.

Compares two pipeline structures side by side using BenchmarkSweep:
  - plain: standard single-pass inference at 640×640
  - tiled: SAHI-style tiling with Scatter/Gather + NMM deduplication

Each config is run with warmup + N measured runs. The resulting table shows
per-operator latency for each variant so you can see exactly where tiling adds
cost (Tile, Scatter/Gather, Stitch, NMM) vs plain inference.

Results are saved as JSON for later comparison or forwarding to MLflow / W&B.

Usage:
    python run_yolo8_benchmark_sweep.py
    python run_yolo8_benchmark_sweep.py --runs 20 --warmup 3
    python run_yolo8_benchmark_sweep.py --model s --save results/
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
from examples.run_yolo8_tile import yolo8_tiled_pipeline

from ml_pipes import Pipeline
from ml_pipes.benchmark import BenchmarkResult, BenchmarkSweep, MeasurementConfig


def _make_pipeline(model_path: Path, output_path: Path, coco_classes: list[str]):
    def factory(config: dict) -> Pipeline:
        if config.get("tiled", False):
            return (
                decode()
                + yolo8_tiled_pipeline(
                    model_path,
                    conf_threshold=config.get("conf_threshold", 0.25),
                    slice_wh=config.get("slice_wh", (320, 320)),
                    overlap_wh=config.get("overlap_wh", (80, 80)),
                    max_concurrency=config.get("max_concurrency", 4),
                )
                + visualize_detections_and_store(output_path, coco_classes)
            )
        return (
            decode()
            + yolo8_inference_pipeline(model_path, conf_threshold=config.get("conf_threshold", 0.25))
            + visualize_detections_and_store(output_path, coco_classes)
        )
    return factory


def _input_fn(image_path: Path, label: str | None = None):
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

    configs = [
        {"tiled": False, "conf_threshold": 0.25},
        {"tiled": True,  "conf_threshold": 0.25, "slice_wh": (320, 320), "overlap_wh": (80, 80)},
        {"tiled": True,  "conf_threshold": 0.25, "slice_wh": (480, 480), "overlap_wh": (80, 80)},
    ]

    config = MeasurementConfig(runs=args.runs, warmup=args.warmup, percentiles=(0.50, 0.95, 0.99))

    _fn = _input_fn(image_path)
    sweep = BenchmarkSweep(
        factory=_make_pipeline(model_path, output_path, COCO_CLASSES),
        configs=configs,
        data_factory=lambda _, fn=_fn: fn,
        data_configs=[{"_label": "coco_sample"}],
        measurement=config,
    )

    print(f"\nRunning {len(configs)} configs "
          f"({args.warmup} warmup + {args.runs} measured each)\n", file=sys.stderr)

    results = sweep.run()
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
