"""
Scripted `BenchmarkBuilder` sweep for plain vs tiled YOLOv8 inference.

This example focuses on comparing two pipeline structures side by side in one
script. It is the best reference for a handpicked config sweep built with
reusable decorated factories plus a plain dynamic input callable.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark_sweep.py --model n --runs 20
    python examples/benchmarks/run_yolo8_benchmark_sweep.py --model s --save results/

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

from ml_pipes import Detections, ImagePayload, Pipeline, pipeline_factory
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


_DEFAULT_MODEL_VARIANT = "n"
_model_name, _model_url = YOLO8_MODELS[_DEFAULT_MODEL_VARIANT]
_DEFAULT_MODEL_PATH = resolve_model_path(ASSETS_DIR, _model_name, _model_url, _DEFAULT_MODEL_VARIANT)
_DEFAULT_OUTPUT_PATH = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, _model_name)


@pipeline_factory
def yolo8_plain_benchmark_pipeline(
    model_path: Path = _DEFAULT_MODEL_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    conf_threshold: float = 0.25,
) -> Pipeline[str | Path, tuple[ImagePayload, Detections]]:
    return (
        decode()
        + yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path = _DEFAULT_MODEL_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    conf_threshold: float = 0.25,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
    max_concurrency: int = 4,
) -> Pipeline[str | Path, tuple[ImagePayload, Detections]]:
    return (
        decode()
        + yolo8_tiled_pipeline(
            model_path,
            conf_threshold=conf_threshold,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            max_concurrency=max_concurrency,
        )
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )


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

    data_input = _input_fn(image_path)

    plain_results = (
        BenchmarkBuilder.factory(yolo8_plain_benchmark_pipeline)
        .pipeline_config(model_path=model_path, output_path=output_path)
        .data_input(data_input)
        .runs(args.runs).warmup(args.warmup)
        .run(verbose=False)
    )

    tiled_results = (
        BenchmarkBuilder.factory(yolo8_tiled_benchmark_pipeline)
        .pipeline_config(model_path=model_path, output_path=output_path)
        .pipeline_config_set([
            {"slice_wh": (320, 320)},
            {"slice_wh": (480, 480), "overlap_wh": (120, 120)},
        ])
        .data_input(data_input)
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
