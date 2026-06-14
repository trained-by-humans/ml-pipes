"""
YOLOv8 tiled detection example.

Tiles the input image into overlapping patches, runs inference on each patch
in parallel via Scatter/Gather, then stitches detections back and suppresses
duplicates with NMM. Useful for high-resolution images where small objects
would be missed at 640×640.

Usage:
    python run_yolo8_tile.py
    python run_yolo8_tile.py --model s --slice-wh 320 320 --overlap-wh 80 80
    python run_yolo8_tile.py --input image.jpg --output result.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    COCO_CLASSES,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    add_assets_dir_arg,
    add_conf_threshold_arg,
    add_model_arg,
    build_output_path,
    decode,
    download_if_missing,
    resolve_model_path,
    visualize_detections_and_store,
)
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline

from ml_pipes import (
    Detections,
    Gather,
    ImagePayload,
    Inline,
    NMM,
    Pick,
    Pipeline,
    Recall,
    Scatter,
    Stitch,
    Store,
    Tile,
)


def yolo8_tiled_pipeline(
    model_path: Path,
    conf_threshold: float = 0.25,
    slice_wh: tuple[int, int] = (200, 200),
    overlap_wh: tuple[int, int] = (40, 40),
    max_concurrency: int = 4,
    iou_threshold: float = 0.4,
) -> Pipeline[ImagePayload, Detections]:
    """Tiled YOLOv8 inference pipeline.

    Tiles → parallel inference per tile → stitch → NMM to suppress cross-tile duplicates.
    """
    return Pipeline([
        Tile(slice_wh=slice_wh, overlap_wh=overlap_wh),
        Store("tile_rects", source=1),
        Pick(0),
        Scatter(max_concurrency=max_concurrency),
        Inline(yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)),
        Gather(),
        Recall("tile_rects"),
        Stitch(),
        NMM(iou_threshold=iou_threshold),
    ], auto_validate=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assets_dir_arg(parser)
    add_model_arg(parser, list(YOLO8_MODELS))
    add_conf_threshold_arg(parser)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--slice-wh", type=int, nargs=2, default=[200, 200], metavar=("W", "H"),
                        help="Tile width and height in pixels (default: 320 320).")
    parser.add_argument("--overlap-wh", type=int, nargs=2, default=[40, 40], metavar=("W", "H"),
                        help="Overlap between tiles in pixels (default: 80 80).")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max parallel tile inference workers (default: 4).")
    args = parser.parse_args()

    assets_dir = args.assets_dir
    model_name, model_url = YOLO8_MODELS[args.model]
    model_path = resolve_model_path(assets_dir, model_name, model_url, args.model)
    if model_path is None:
        return 1

    if args.input is not None:
        image_path = args.input
        if not image_path.exists():
            print(f"Error: input file not found: {image_path}", file=sys.stderr)
            return 1
    else:
        image_path = assets_dir / COCO_IMAGE_NAME
        print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
        download_if_missing(COCO_IMAGE_URL, image_path)

    output_path = args.output or build_output_path(assets_dir, image_path.name, model_name.replace(".onnx", "_tile.onnx"))

    pipeline = (
        decode()
        + yolo8_tiled_pipeline(
            model_path,
            conf_threshold=args.conf_threshold,
            slice_wh=(args.slice_wh[0], args.slice_wh[1]),
            overlap_wh=(args.overlap_wh[0], args.overlap_wh[1]),
            max_concurrency=args.max_concurrency,
        )
        + visualize_detections_and_store(output_path, COCO_CLASSES)
    )
    pipeline.validate()
    pipeline.describe()
    pipeline(image_path)
    print(f"Output written to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
