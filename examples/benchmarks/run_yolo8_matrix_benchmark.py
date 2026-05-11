"""
Matrix benchmark example: sweep batch_size × serialize axes for the batched YOLOv8 pipeline.

BenchmarkMatrix expands the cartesian product of all axes automatically and
delegates each cell to BenchmarkSweep → Benchmark. Mirrors benchmark_batch.py
but uses the ml-pipes benchmarking layer for per-operator latency breakdown
instead of raw wall-clock throughput.

Each cell runs a fresh pipeline (batch_size, serialize) with warmup + N measured
single-threaded requests. Because calls are sequential, Batch always fires via
timeout (batch of 1 per request). This measures per-request latency through the
full operator chain — a different view from the concurrent throughput in
benchmark_batch.py.

Axes swept:
  batch_size  — controls the Batch operator buffer size
  serialize   — controls whether concurrent session.run() calls are serialized

Usage:
    python run_yolo8_matrix_benchmark.py
    python run_yolo8_matrix_benchmark.py --runs 20 --warmup 3
    python run_yolo8_matrix_benchmark.py --save results/
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
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    download_if_missing,
)
from examples.run_batch_yolo8_onnx import MODEL_NAME, _export_dynamic_model, build_pipeline

from ml_pipes import Pipeline
from ml_pipes.benchmark import BenchmarkMatrix, MeasurementConfig


def _make_pipeline(model_path: Path):
    def factory(config: dict) -> Pipeline:
        return build_pipeline(
            model_path,
            batch_size=config["batch_size"],
            timeout=0.05,
            serialize=config["serialize"],
        )
    return factory


def _input_fn(image_path: Path):
    def fn():
        return (image_path.name, image_path, None, None)
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per cell (default: 20).")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per cell (default: 3).")
    parser.add_argument("--save", type=Path, default=None, help="Directory to save per-cell result JSON files.")
    args = parser.parse_args()

    assets_dir: Path = args.assets_dir
    model_path = assets_dir / MODEL_NAME

    if not model_path.exists():
        print(f"Exporting YOLOv8n nano (dynamic batch) → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    axes = {
        "batch_size": [1, 2, 4, 8],
        "serialize":  [True, False],
    }
    total_cells = 1
    for v in axes.values():
        total_cells *= len(v)

    config = MeasurementConfig(runs=args.runs, warmup=args.warmup, percentiles=(0.50, 0.95, 0.99))

    print(
        f"\nAxes: { {k: len(v) for k, v in axes.items()} } → {total_cells} cells"
        f" ({args.warmup} warmup + {args.runs} measured each)\n",
        file=sys.stderr,
    )

    matrix = BenchmarkMatrix(
        pipeline_factory=_make_pipeline(model_path),
        axes=axes,
        inputs=[_input_fn(image_path)],
        config=config,
    )

    results = matrix.run()
    print(BenchmarkMatrix.to_table(results))

    if args.save:
        save_dir = args.save
        save_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            result.save(str(save_dir / result.slug(".json")))
        print(f"\nResults saved to {save_dir}/", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
