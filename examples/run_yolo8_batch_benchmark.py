from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from common import add_assets_dir_arg, COCO_IMAGE_NAME, COCO_IMAGE_URL, download_if_missing
from run_yolo8_batch import MODEL_NAME, _export_dynamic_model, build_pipeline


# Throughput benchmark for the Batch/UnBatch pipeline.
#
# Each dimension (batch_size, workers, lock) accepts one or more values;
# the script runs the full cartesian product and prints one row per config.
#
# Usage:
#   python run_yolo8_batch_benchmark.py                         # defaults
#   python run_yolo8_batch_benchmark.py --batch-size 4 8 --workers 8 --lock on off
#   python run_yolo8_batch_benchmark.py --batch-size 1 2 4 8 --workers 1 4 8 --lock on off
#
# Options:
#   --batch-size  INT [INT ...]   batch sizes to sweep   (default: 1 2 4 8)
#   --workers     INT [INT ...]   thread-pool sizes       (default: 1 4 8)
#   --lock        on|off [...]    serialize inference?    (default: on)
#   --images      N               images per run          (default: 16)
#   --repeats     N               runs per config, median (default: 3)
#   --assets-dir  PATH


def _run_once(pipeline, image_paths: list[Path], workers: int) -> float:
    """Return wall-clock seconds to process all images with the given worker count."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(pipeline, p) for p in image_paths]
        wait(futures)
    elapsed = time.perf_counter() - t0
    for f in futures:
        f.result()  # re-raise any exception
    return elapsed


def _measure(model_path: Path, batch_size: int, workers: int, serialize: bool,
             image_paths: list[Path], repeats: int) -> float:
    """Build a fresh pipeline for the config and return median wall time."""
    pipeline = build_pipeline(model_path, batch_size=batch_size, timeout=0.05,
                              serialize=serialize)
    return statistics.median(
        _run_once(pipeline, image_paths, workers) for _ in range(repeats)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pipeline throughput benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=[1, 2, 4, 8],
        metavar="N", help="Batch sizes to sweep (default: 1 2 4 8).",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[4, 8, 16],
        metavar="N", help="Thread-pool sizes to sweep (default: 4, 8, 16).",
    )
    parser.add_argument(
        "--lock", nargs="+", choices=["on", "off"], default=["off"],
        metavar="on|off", help="Inference lock values to sweep (default: off).",
    )
    parser.add_argument(
        "--images", type=int, default=16, metavar="N",
        help="Number of images per run (default: 16).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, metavar="N",
        help="Repetitions per config — median is reported (default: 3).",
    )
    add_assets_dir_arg(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME

    if not model_path.exists():
        print(f"Exporting YOLOv8n nano (dynamic batch) → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

    sample = assets_dir / COCO_IMAGE_NAME
    download_if_missing(COCO_IMAGE_URL, sample)
    image_paths = [sample] * args.images

    configs: list[tuple[int, int, str]] = list(
        itertools.product(args.batch_size, args.workers, args.lock)
    )

    print(
        f"Benchmarking {len(configs)} configs"
        f" | {args.images} images/run"
        f" | {args.repeats} repeats (median reported)",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # --- run ---
    Result = tuple[int, int, str, float]  # (batch, workers, lock, wall_s)
    results: list[Result] = []

    for batch_size, workers, lock in configs:
        serialize = lock == "on"
        print(
            f"  batch={batch_size} workers={workers} lock={lock} ...",
            end="  ", flush=True, file=sys.stderr,
        )
        wall = _measure(model_path, batch_size, workers, serialize, image_paths, args.repeats)
        results.append((batch_size, workers, lock, wall))
        print(f"{wall:.3f}s", file=sys.stderr)

    print(file=sys.stderr)

    # --- baseline: (batch=1, workers=1, lock=on) if present, else first result ---
    baseline_wall = next(
        (w for b, wk, l, w in results if b == 1 and wk == 1 and l == "on"),
        results[0][3],
    )

    # --- table ---
    header = (f"  {'batch':>5}  {'workers':>7}  {'lock':>4}"
              f"  {'ms/image':>8}  {'wall':>12}  {'vs serial':>9}")
    sep = (f"  {'─' * 5}  {'─' * 7}  {'─' * 4}"
           f"  {'─' * 8}  {'─' * 12}  {'─' * 9}")

    print(header)
    print(sep)
    for batch_size, workers, lock, wall in results:
        ms_per = wall / args.images * 1000
        speedup = baseline_wall / wall
        is_base = (batch_size == 1 and workers == 1 and lock == "on")
        marker = " <-- baseline" if is_base else ""
        print(
            f"  {batch_size:>5}  {workers:>7}  {lock:>4}"
            f"  {ms_per:>8.1f}  {wall:>10.3f} s  {speedup:>8.2f} x{marker}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
