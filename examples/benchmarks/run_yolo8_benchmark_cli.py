"""
CLI benchmark target for `python -m ml_pipes benchmark`.

Unlike the script-based examples, this module is meant to be discovered by the
benchmark CLI. It shows the reusable module-level factory pattern:
`@pipeline_factory` for the pipeline and `@data_factory` for the input source.

Run from the repo root:
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --arg slice_wh=480x480 --runs 20 --warmup 3
    python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
        --arg model_path=path/to/model.onnx

See `docs/BENCHMARKING.md` for sweep variants, shared CLI options, and factory
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
from examples.run_yolo8_onnx import BUNDLED_MODEL_PATH

from ml_pipes.core import Pipeline
from ml_pipes.factory import (
    pipeline_factory,
    data_factory,
)
from ml_pipes.vision import (
    Detections,
    ImagePayload,
)
from ml_pipes.benchmark import InputFn


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: Path | None = None,
    output_path: Path | None = None,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
    conf_threshold: float = 0.25,
    max_concurrency: int = 4,
) -> Pipeline[str | Path, tuple[ImagePayload, Detections]]:
    """Tiled YOLOv8 pipeline — the target pipeline for CLI benchmarking."""
    from examples.run_yolo8_tile import yolo8_tiled_pipeline
    resolved_model_path = resolve_model_path(model_path, BUNDLED_MODEL_PATH)
    resolved_output_path = output_path or build_output_path(ASSETS_DIR, "run_yolo8_benchmark_cli_tiled.jpg", resolved_model_path.name)
    return (
        decode()
        + yolo8_tiled_pipeline(
            resolved_model_path,
            conf_threshold=conf_threshold,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            max_concurrency=max_concurrency,
        )
        + visualize_detections_and_store(resolved_output_path, COCO_CLASSES)
    )


@data_factory
def coco_sample_input(
    image_path: Path = ASSETS_DIR / COCO_IMAGE_NAME,
) -> InputFn:
    """Downloads the standard COCO sample image if needed and returns an InputFn."""
    download_if_missing(COCO_IMAGE_URL, image_path)
    def fn():
        return (image_path.name, image_path, None, None)
    return fn
