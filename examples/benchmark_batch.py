from __future__ import annotations

# Throughput benchmark for the Batch/UnBatch pipeline.
#
# Sweeps a matrix of (batch_size, workers) configurations, runs each one
# several times, and prints a table like:
#
#   batch  workers  ms/image  wall (8 imgs)  vs serial
#   -----  -------  --------  ------------  ---------
#       1        1    124.2      0.994 s      1.00 x
#       1        8     33.6      0.269 s      3.70 x
#       2        8     31.7      0.254 s      3.92 x
#       4        8     40.2      0.321 s      3.09 x
#       8        8     50.0      0.400 s      2.49 x
#
# Usage:
#   python benchmark_batch.py
#
# Options:
#   --images N          number of images to process per run (default: 8)
#   --repeats N         repetitions per config — median is reported (default: 3)
#   --assets-dir PATH

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from common import COCO_IMAGE_NAME, COCO_IMAGE_URL, download_if_missing
from run_batch_yolo8n_onnx import ASSETS_DIR, MODEL_NAME, _export_dynamic_model, build_pipeline

# (batch_size, workers) pairs to sweep.
CONFIGS: list[tuple[int, int]] = [
    (1, 1),
    (1, 4),
    (2, 4),
    (4, 4),
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 8),
]


def _run_once(pipeline, image_paths: list[Path]) -> float:
    """Return wall-clock seconds to process all images through the pipeline."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
        futures = [pool.submit(pipeline, p) for p in image_paths]
        wait(futures)
    elapsed = time.perf_counter() - t0
    for f in futures:
        f.result()  # re-raise any exception
    return elapsed


def _measure(model_path: Path, batch_size: int, workers: int,
             image_paths: list[Path], repeats: int, serialize: bool) -> float:
    """Build a fresh pipeline for the config and return median wall time."""
    pipeline = build_pipeline(model_path, batch_size=batch_size, timeout=0.05,
                              serialize=serialize)
    samples: list[float] = []
    for _ in range(repeats):
        samples.append(_run_once(pipeline, image_paths))
    return statistics.median(samples)


def _fmt_row(batch: int, workers: int, ms_per: float, wall: float,
             speedup: float, is_baseline: bool) -> str:
    marker = " <-- baseline" if is_baseline else ""
    return (
        f"  {batch:>5}  {workers:>7}  {ms_per:>8.1f}  {wall:>10.3f} s  {speedup:>8.2f} x{marker}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pipeline throughput benchmark.",
    )
    parser.add_argument(
        "--images",
        type=int,
        default=16,
        metavar="N",
        help="Number of images per run (default: 8).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        metavar="N",
        help="Repetitions per config — median is reported (default: 3).",
    )
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument(
        "--no-serialize",
        action="store_true",
        help="Disable the inference lock (allows concurrent session.run() calls).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME

    if not model_path.exists():
        print(f"Exporting YOLOv8n (dynamic batch) → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

    sample = assets_dir / COCO_IMAGE_NAME
    download_if_missing(COCO_IMAGE_URL, sample)
    image_paths = [sample] * args.images

    serialize = not args.no_serialize
    print(
        f"Benchmarking {len(CONFIGS)} configs"
        f" | {args.images} images/run"
        f" | {args.repeats} repeats (median reported)"
        f" | serialize={'on' if serialize else 'off'}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    header = f"  {'batch':>5}  {'workers':>7}  {'ms/image':>8}  {'wall':>12}  {'vs serial':>9}"
    sep    = f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*12}  {'─'*9}"

    results: list[tuple[int, int, float]] = []  # (batch, workers, wall_s)

    for batch_size, workers in CONFIGS:
        print(f"  running batch={batch_size} workers={workers} ...", end="  ", flush=True, file=sys.stderr)
        wall = _measure(model_path, batch_size, workers, image_paths, args.repeats,
                        serialize=serialize)
        results.append((batch_size, workers, wall))
        print(f"{wall:.3f}s", file=sys.stderr)

    print(file=sys.stderr)

    serial_wall = next(w for b, wk, w in results if b == 1 and wk == 1)

    print(header)
    print(sep)
    for batch_size, workers, wall in results:
        ms_per = wall / args.images * 1000
        speedup = serial_wall / wall
        is_baseline = (batch_size == 1 and workers == 1)
        print(_fmt_row(batch_size, workers, ms_per, wall, speedup, is_baseline))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
