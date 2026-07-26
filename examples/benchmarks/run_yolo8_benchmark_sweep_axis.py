"""
Axis-sweep example for tile size and overlap on the tiled YOLOv8 pipeline.

Unlike the plain-vs-tiled sweep example, this script focuses on cartesian axis
expansion plus filtering within one pipeline template.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark_sweep_axis.py --model n --runs 20
    python examples/benchmarks/run_yolo8_benchmark_sweep_axis.py --model s --save results/

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
from examples.benchmarks.benchmark_common import YOLO8_MODELS, resolve_model_variant_path
from examples.run_yolo8_tile import yolo8_tiled_pipeline

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.vision import (
    Detections,
    ImagePayload,
)
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


_DEFAULT_MODEL_VARIANT = "n"
_model_name, _model_url = YOLO8_MODELS[_DEFAULT_MODEL_VARIANT]
_DEFAULT_MODEL_PATH = resolve_model_variant_path(_model_name, _model_url, _DEFAULT_MODEL_VARIANT)
_DEFAULT_OUTPUT_PATH = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, _model_name)


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path = _DEFAULT_MODEL_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
) -> Pipeline[str | Path, tuple[ImagePayload, Detections]]:
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

    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_variant_path(model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)

    output_path = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, model_name)

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
