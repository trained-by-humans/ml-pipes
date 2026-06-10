"""
CLI benchmark target for `python -m ml_pipes benchmark`.

Unlike the script-based examples, this module is meant to be discovered by the
benchmark CLI. It shows the reusable module-level factory pattern:
`@pipeline_factory` for the pipeline and `@data_factory` for the input source.

Run from the repo root:
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --arg slice_wh=480x480 --runs 20 --warmup 3

See `BENCHMARKING.md` for sweep variants, shared CLI options, and factory
rules.
"""
from __future__ import annotations

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
    resolve_model_path,
    visualize_detections_and_store,
)
from examples.run_yolo8_onnx import YOLO8_MODELS

from ml_pipes import Pipeline, pipeline_factory, data_factory
from ml_pipes.benchmark import InputFn


_DEFAULT_MODEL_VARIANT = "n"
_model_name, _model_url = YOLO8_MODELS[_DEFAULT_MODEL_VARIANT]
_DEFAULT_MODEL_PATH = resolve_model_path(ASSETS_DIR, _model_name, _model_url, _DEFAULT_MODEL_VARIANT)
_DEFAULT_IMAGE_PATH = ASSETS_DIR / COCO_IMAGE_NAME
_DEFAULT_OUTPUT_PATH = build_output_path(ASSETS_DIR, COCO_IMAGE_NAME, _model_name)


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path = _DEFAULT_MODEL_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
    conf_threshold: float = 0.25,
    max_concurrency: int = 4,
) -> Pipeline:
    """Tiled YOLOv8 pipeline — the target pipeline for CLI benchmarking."""
    from examples.run_yolo8_tile import yolo8_tiled_pipeline
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


@data_factory
def coco_sample_input(
    image_path: Path = _DEFAULT_IMAGE_PATH,
) -> InputFn:
    """Downloads the standard COCO sample image if needed and returns an InputFn."""
    download_if_missing(COCO_IMAGE_URL, image_path)
    def fn():
        return (image_path.name, image_path, None, None)
    return fn
