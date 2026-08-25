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
    resolve_input_path,
    resolve_model_path,
    visualize_detections_and_store,
)
from examples.run_yolo8_onnx import BUNDLED_MODEL_PATH

from ml_pipes.core import Pipeline
from ml_pipes.factory import (
    pipeline_factory,
    data_factory,
)
from ml_pipes.benchmark import InputFn
from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import ImagePayload


@pipeline_factory
def yolo8_tiled_benchmark_pipeline(
    model_path: str | None = None,
    output_path: str | None = None,
    slice_wh: tuple[int, int] = (320, 320),
    overlap_wh: tuple[int, int] = (80, 80),
    conf_threshold: float = 0.25,
    max_concurrency: int = 4,
) -> Pipeline[str | Path, tuple[ImagePayload, TensorRegistry]]:
    """Tiled YOLOv8 pipeline — the target pipeline for CLI benchmarking."""
    from examples.run_yolo8_tile import yolo8_tiled_pipeline

    resolved_model_path = resolve_model_path(
        Path(model_path) if model_path is not None else None,
        BUNDLED_MODEL_PATH,
    )
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else build_output_path(
            ASSETS_DIR,
            "run_yolo8_benchmark_cli_tiled.jpg",
            resolved_model_path.name,
        )
    )
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
    image_path: str | None = None,
) -> InputFn:
    """Downloads the standard COCO sample image if needed and returns an InputFn."""
    resolved_image_path = resolve_input_path(
        Path(image_path) if image_path is not None else None,
        ASSETS_DIR / COCO_IMAGE_NAME,
        COCO_IMAGE_URL,
    )

    def fn():
        return (resolved_image_path.name, resolved_image_path, None, None)

    return fn
