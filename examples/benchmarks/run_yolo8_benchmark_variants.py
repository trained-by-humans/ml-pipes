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
    resolve_input_path,
    resolve_model_path,
    visualize_detections_and_store,
)

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.onnx import (
    Extract,
    Infer,
)
from ml_pipes.standard import (
    Pick,
    Recall,
    Store,
)
from ml_pipes.tensor import (
    ArgMax,
    GatherScores,
    Slice,
    Squeeze,
    TensorRegistry,
    Transpose,
)
from ml_pipes.vision import (
    ConvertBoxFormat,
    ImagePayload,
    NMS,
    Normalize,
    ProjectBoxes,
    Resize,
)
from ml_pipes.benchmark import BenchmarkBuilder, BenchmarkResult


YOLO8_MODELS: dict[str, tuple[str, str | None]] = {
    "n": ("yolov8n_variants.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"),
    "s": ("yolov8s_variants.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"),
    "m": ("yolov8m_variants.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8m.onnx"),
    "l": ("yolov8l_variants.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8l.onnx"),
    "x": ("yolov8x_variants.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8x.onnx"),
}


def yolo8_variant_inference_pipeline(
    model_path: Path,
    conf_threshold: float = 0.25,
) -> Pipeline[ImagePayload, TensorRegistry]:
    return Pipeline(
        [
            Resize((640, 640)),
            Store("resize_transform", source=1),
            Pick(0),
            Normalize(),
            Infer(model_path),
            Extract("predictions", as_="preds"),
            Squeeze("preds"),
            Transpose("preds"),
            Slice("preds", slice(None, 4), as_="boxes"),
            Slice("preds", slice(4, None), as_="scores"),
            ArgMax("scores", as_="classes"),
            GatherScores("scores", "classes"),
            ConvertBoxFormat(from_="cxcywh"),
            NMS(conf_threshold=conf_threshold),
            Recall("resize_transform"),
            ProjectBoxes(),
        ],
        auto_validate=True,
    )


@pipeline_factory
def yolo8_variant_pipeline(
    model_path: Path | None = None,
    output_path: Path | None = None,
    conf_threshold: float = 0.25,
) -> Pipeline[str | Path, tuple[ImagePayload, TensorRegistry]]:
    default_model_name, default_model_url = YOLO8_MODELS["n"]
    resolved_model_path = resolve_model_path(model_path, ASSETS_DIR / default_model_name, default_model_url)
    resolved_output_path = output_path or build_output_path(ASSETS_DIR, "run_yolo8_benchmark_variants.jpg", resolved_model_path.name)
    return (
        decode()
        + yolo8_variant_inference_pipeline(resolved_model_path, conf_threshold=conf_threshold)
        + visualize_detections_and_store(resolved_output_path, COCO_CLASSES)
    )


def _try_resolve_model_variant(variant: str) -> tuple[str, Path] | None:
    model_name, model_url = YOLO8_MODELS[variant]
    model_path = ASSETS_DIR / model_name
    if model_path.exists():
        return model_name, model_path
    if model_url is None:
        print(f"warning: no download URL configured for variant {variant!r}, skipping", file=sys.stderr)
        return None
    try:
        download_if_missing(model_url, model_path)
    except Exception as exc:
        print(f"warning: could not resolve variant {variant!r} ({model_name}): {exc}", file=sys.stderr)
        return None
    if not model_path.exists():
        print(f"warning: model file not found for variant {variant!r}: {model_path}", file=sys.stderr)
        return None
    return model_name, model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants", nargs="+", default=["n", "s"],
        metavar="VARIANT", help="Model variants to benchmark (default: n s).",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per variant (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per variant (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-variant result JSON files.")
    args = parser.parse_args()

    image_path = resolve_input_path(None, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    input_fn = lambda: (image_path.name, image_path, None, None)

    configs = []
    for variant in args.variants:
        if variant not in YOLO8_MODELS:
            print(f"warning: unknown variant {variant!r}, skipping", file=sys.stderr)
            continue
        resolved = _try_resolve_model_variant(variant)
        if resolved is None:
            continue
        _, model_path = resolved
        configs.append({"model_path": model_path})

    if not configs:
        print("error: no model variants available — download at least one model file first", file=sys.stderr)
        return 1

    results = (
        BenchmarkBuilder.factory(yolo8_variant_pipeline)
        .pipeline_config_set(configs)
        .data_input(input_fn)
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
