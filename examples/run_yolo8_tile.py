"""
YOLOv8 tiled detection on a sample image.

This example slices the image into overlapping tiles, runs inference per tile,
and merges the detections with NMM.

Run from the repo root:
    python examples/run_yolo8_tile.py
    python examples/run_yolo8_tile.py --input path/to/photo.jpg --output result.jpg
    python examples/run_yolo8_tile.py --model-path path/to/model.onnx --slice-wh 320 320 --overlap-wh 80 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
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
from run_yolo8_onnx import BUNDLED_MODEL_PATH, yolo8_inference_pipeline

from ml_pipes.core import (
    Inline,
    Pipeline,
)
from ml_pipes.standard import (
    Gather,
    Pick,
    Recall,
    Scatter,
    Store,
)
from ml_pipes.vision import (
    Detections,
    ImagePayload,
    NMM,
    Stitch,
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
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to the bundled YOLOv8n model in the assets directory.")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Minimum confidence score for detections (default: 0.25).")
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path. Defaults to a file under the assets directory.")
    parser.add_argument("--slice-wh", type=int, nargs=2, default=[200, 200], metavar=("W", "H"),
                        help="Tile width and height in pixels (default: 200 200).")
    parser.add_argument("--overlap-wh", type=int, nargs=2, default=[40, 40], metavar=("W", "H"),
                        help="Overlap between tiles in pixels (default: 40 40).")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max parallel tile inference workers (default: 4).")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
    tiled_model_name = f"{model_path.stem}_tile{model_path.suffix}"
    output_path = args.output or build_output_path(ASSETS_DIR, image_path.name, tiled_model_name)

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
