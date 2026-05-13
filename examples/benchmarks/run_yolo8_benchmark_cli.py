"""
CLI benchmark example: YOLOv8 tiled pipeline annotated for python -m ml_pipes.

This module demonstrates how to expose a pipeline and its input for benchmarking
directly from the command line, without writing a benchmark script.

The two decorators do all the work:
  @pipeline_factory  — marks the pipeline constructor; CLI calls it with config kwargs
  @data_factory      — marks the input driver; CLI calls it to get an InputFn

Usage (single run, with optional config overrides):
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --arg slice_wh=480x480 --runs 20 --warmup 3

Usage (sweep over explicit configs):
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --config '{"slice_wh":[320,320],"overlap_wh":[80,80]}' \
        --config '{"slice_wh":[480,480],"overlap_wh":[80,80]}' \
        --runs 20 --warmup 3

Usage (axis sweep — cartesian product of axes):
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --axis slice_wh=240x240,320x320,480x480 \
        --axis overlap_wh=40x40,80x80 \
        --runs 20 --warmup 3

All parameters without defaults (model_path, output_path, image_path) must be
supplied via --config or --axis when running from the CLI.  The defaults below
are set up for running from the examples/ directory with the standard asset cache.
"""
from __future__ import annotations

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
