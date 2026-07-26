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
    build_output_path,
    decode,
    resolve_input_path,
    visualize_detections_and_store,
)
from examples.benchmarks.benchmark_common import YOLO8_MODELS, resolve_model_variant
from examples.run_yolo8_onnx import yolo8_inference_pipeline
from examples.run_yolo8_tile import yolo8_tiled_pipeline

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.vision import (
    Detections,
    ImagePayload,
)
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


_DEFAULT_MODEL_VARIANT = "n"
_DEFAULT_MODEL_NAME, _DEFAULT_MODEL_PATH = resolve_model_variant(_DEFAULT_MODEL_VARIANT)
_DEFAULT_OUTPUT_PATH = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, _DEFAULT_MODEL_NAME)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(YOLO8_MODELS),
        default="n",
        help=f"Model variant ({' → '.join(YOLO8_MODELS)}).",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per cell (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per cell (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-cell result JSON files.")
    args = parser.parse_args()

    model_name, model_path = resolve_model_variant(args.model)

    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)

    output_path = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, model_name)

    input_fn = lambda: (image_path.name, image_path, None, None)

    plain_results = (
        BenchmarkBuilder.factory(yolo8_plain_benchmark_pipeline)
        .pipeline_config(model_path=model_path, output_path=output_path)
        .data_input(input_fn)
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
