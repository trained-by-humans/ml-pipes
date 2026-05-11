"""
Inspection example: visualise the data at every pipeline step.

Three pipelines are demonstrated:

  simple   — single image through the full YOLOv8 detection pipeline.

  batched  — 8 image paths in, Scatter decodes them concurrently,
             Batch groups them into batches of 4 for inference,
             UnBatch/Gather collect per-image detections.

  tiled    — single image tiled into overlapping 240×240 patches,
             each tile inferred independently via Scatter, results
             stitched back and deduplicated with NMM.

Usage:
    # Open result in the default web browser (default behaviour)
    python run_inspect.py

    # Use the tiled pipeline
    python run_inspect.py --pipeline tiled

    # Use the batched pipeline
    python run_inspect.py --pipeline batched

    # Save HTML report to a file (does not open browser)
    python run_inspect.py --save report.html

    # Save a matplotlib figure to a PNG
    python run_inspect.py --plot inspect.png

    # Serialize result to disk for later analysis
    python run_inspect.py --dump result.pkl

    # Load a previously serialized result and open it in the browser
    python run_inspect.py --load result.pkl

    # Print step labels and shapes to stdout only
    python run_inspect.py --print-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_model_arg,
    download_if_missing,
    resolve_model_path,
    decode,
)
from common import COCO_CLASSES, build_output_path, visualize_detections_and_store
from ml_pipes import DrawBoxes, SaveImage
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from run_yolo8_batch import MODEL_NAME as BATCH_MODEL_NAME, build_pipeline as build_batch_pipeline
from ml_pipes import (
    Gather,
    Inline,
    InspectionResult,
    InspectionSerializer,
    NMM,
    Pick,
    Pipeline,
    PipelineInspector,
    Recall,
    Scatter,
    Stitch,
    Store,
    Tile,
)


# ---------------------------------------------------------------------------
# Inspection logic
# ---------------------------------------------------------------------------

def run_inspection_simple(model_path: Path, image_path: Path, output_path: Path) -> InspectionResult:
    """Single image → decode → infer → visualize."""
    pipeline: Pipeline = (
        decode()
        + yolo8_inference_pipeline(model_path)
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline.validate()
    print("Running simple inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)
    print(result)
    return result


def run_inspection_batched(assets_dir: Path, image_path: Path) -> InspectionResult:
    """8 paths → Scatter(decode+preprocess) → Gather → Batch(4) Infer → UnBatch → Detections.

    Uses the dynamic-batch ONNX model (yolov8n_dynamic.onnx) which accepts any
    batch size. Scatter fans the list out to worker threads; Gather collects
    preprocessed tensors; the Batch region runs batched inference.
    """
    from run_yolo8_batch import _export_dynamic_model

    model_path = assets_dir / BATCH_MODEL_NAME
    if not model_path.exists():
        print(f"Exporting dynamic-batch model → {model_path}", file=sys.stderr)
        _export_dynamic_model(model_path)

    # build_batch_pipeline expects single-threaded Batch gate usage;
    # wrap it with Scatter/Gather so inspect() can drive it from a list.
    per_image = build_batch_pipeline(model_path, batch_size=4, timeout=1.0)
    pipeline = Pipeline([
        Scatter(max_concurrency=4),
        Inline(per_image),
        Gather(),
    ])

    image_paths = [image_path] * 8
    print("Running batched inspection...", file=sys.stderr)
    result = pipeline.inspect(image_paths)
    print(result)
    return result


def run_inspection_tiled(model_path: Path, image_path: Path, output_path: Path) -> InspectionResult:
    """Single image → Tile → Scatter(infer per tile) → Gather → Stitch → NMM → DrawBoxes."""
    infer = yolo8_inference_pipeline(model_path)
    pipeline = Pipeline([
        Inline(decode()),
        Tile(slice_wh=(200, 200), overlap_wh=(40, 40)),
        Store("tile_rects", index=1),
        Pick(0),
        Scatter(max_concurrency=4),
        Inline(infer),
        Gather(),
        Recall("tile_rects"),
        Stitch(),
        NMM(iou_threshold=0.4),
        Recall("source_image", index=0),
        DrawBoxes(class_names=COCO_CLASSES),
        SaveImage(output_path, at=0),
        Pick(0),
    ])
    print("Running tiled inspection...", file=sys.stderr)
    result = pipeline.inspect(image_path)
    print(result)
    return result


def show_result(result: InspectionResult, args: argparse.Namespace) -> None:
    inspector = PipelineInspector()
    if args.save:
        saved = inspector.save_to_html(result, args.save)
        print(f"Inspection report saved to: {saved}", file=sys.stderr)
    elif args.plot:
        saved = inspector.save_to_plot(result, args.plot)
        print(f"Plot saved to: {saved}", file=sys.stderr)
    elif args.dump:
        saved = InspectionSerializer().dump(result, args.dump)
        print(f"Inspection result serialized to: {saved}", file=sys.stderr)
    elif args.print_only:
        pass
    else:
        inspector.show_in_browser(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
    parser.add_argument("--pipeline", choices=["simple", "batched", "tiled"], default="simple",
                        help="Which pipeline to inspect (default: simple).")
    parser.add_argument("--load", metavar="PATH", type=Path, default=None,
                        help="Load a previously serialized result instead of running the pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="PATH", type=Path, default=None,
                       help="Save HTML report to PATH instead of opening a browser.")
    group.add_argument("--plot", metavar="PATH", type=Path, default=None,
                       help="Save a matplotlib figure to PATH (e.g. inspect.png).")
    group.add_argument("--dump", metavar="PATH", type=Path, default=None,
                       help="Serialize the InspectionResult to PATH for later analysis.")
    group.add_argument("--print-only", action="store_true",
                       help="Print step labels and shapes to stdout; do not open any window.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.load:
        result = InspectionResult.load(args.load)
        print(result)
        show_result(result, args)
        return 0

    assets_dir: Path = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    output_path = build_output_path(assets_dir, COCO_IMAGE_NAME, model_name)
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    image_path = assets_dir / COCO_IMAGE_NAME
    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    if args.pipeline == "batched":
        result = run_inspection_batched(assets_dir, image_path)
    elif args.pipeline == "tiled":
        result = run_inspection_tiled(model_path, image_path, output_path)
    else:
        result = run_inspection_simple(model_path, image_path, output_path)
    show_result(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
