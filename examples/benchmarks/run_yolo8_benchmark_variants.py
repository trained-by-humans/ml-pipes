"""
Variant sweep example for YOLOv8 model sizes.

Unlike the other sweep examples, this script keeps the pipeline structure the
same and only varies the model artifact behind it.

Run from the repo root:
    python examples/benchmarks/run_yolo8_benchmark_variants.py --variants n s --runs 20
    python examples/benchmarks/run_yolo8_benchmark_variants.py --variants n s m --save results/

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
    download_if_missing,
    visualize_detections_and_store,
)
from examples.benchmarks.benchmark_common import YOLO8_MODELS, resolve_model_variant_path
from examples.run_yolo8_onnx import yolo8_inference_pipeline

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.vision import (
    Detections,
    ImagePayload,
)
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


@pipeline_factory
def yolo8_variant_pipeline(
    model_path: Path = ASSETS_DIR / "yolov8n.onnx",
    output_path: Path = ASSETS_DIR / "out.jpg",
    conf_threshold: float = 0.25,
) -> Pipeline[str | Path, tuple[ImagePayload, Detections]]:
    return (
        decode()
        + yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )


def _input_fn(image_path: Path):
    def fn():
        return image_path.name, image_path, None, None
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory used to cache downloaded models and sample assets.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(YOLO8_MODELS),
        metavar="VARIANT", help=f"Model variants to benchmark (default: {' '.join(YOLO8_MODELS)}).",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per variant (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per variant (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-variant result JSON files.")
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    configs = []
    for variant in args.variants:
        if variant not in YOLO8_MODELS:
            print(f"warning: unknown variant {variant!r}, skipping", file=sys.stderr)
            continue
        model_name, model_url = YOLO8_MODELS[variant]
        model_path = resolve_model_variant_path(assets_dir, model_name, model_url, variant)
        if model_path is None:
            print(f"warning: model file for variant {variant!r} not found, skipping", file=sys.stderr)
            continue
        output_path = build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)
        configs.append({"model_path": model_path, "output_path": output_path})

    if not configs:
        print("error: no model variants available — download at least one model file first", file=sys.stderr)
        return 1

    results = (
        BenchmarkBuilder.factory(yolo8_variant_pipeline)
        .pipeline_config_set(configs)
        .data_input(_input_fn(image_path))
        .runs(args.runs)
        .warmup(args.warmup)
        .run(verbose=True)
    )
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
