from __future__ import annotations

# Minimal YOLOv8n inference endpoint.
#
# Requires Flask:
#   pip install flask
#
# Start the server:
#   python serve_yolo8n_onnx.py
#
# Run a test call (downloads the sample COCO image if needed):
#   python serve_yolo8n_onnx.py --call
#
# Or send a specific image:
#   python serve_yolo8n_onnx.py --call --input photo.jpg
#
# Or with curl:
#   curl -s -X POST http://localhost:5000/detect \
#        -H "Content-Type: application/octet-stream" \
#        --data-binary @.example_assets/coco_000000039769.jpg | python -m json.tool

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from ml_pipes import (
    ArgMax,
    ConvertBoxFormat,
    Decode,
    GatherScores,
    MapToObjects,
    Infer,
    NMS,
    Normalize,
    Pick,
    Pipeline,
    ProjectBoxes,
    Recall,
    Resize,
    Extract,
    Slice,
    Squeeze,
    Store,
    ToDetections,
    Transpose,
)
from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    download_if_missing,
)


MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MODEL_NAME = "yolov8n.onnx"
HOST = "localhost"
PORT = 5000


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Decode(),
        Resize((640, 640)),
        Store("resize_transform", index=1),
        Pick(0),
        Normalize(),
        Infer(model_path),
        Extract("output0", as_="preds"),
        Squeeze("preds"),
        Transpose("preds"),
        Slice("preds", slice(None, 4), as_="boxes"),
        Slice("preds", slice(4, None), as_="scores"),
        ArgMax("scores", as_="classes"),
        GatherScores("scores", "classes"),
        ConvertBoxFormat(from_="cxcywh"),
        NMS(),
        Recall("resize_transform"),
        ProjectBoxes(),
        ToDetections(),
        MapToObjects(fields={"box": "boxes", "score": "scores", "class_id": "classes"}),
    ])


def run_server(model_path: Path) -> None:
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        print("Flask is required: pip install flask", file=sys.stderr)
        raise SystemExit(1)

    pipeline = build_pipeline(model_path)
    app = Flask(__name__)

    @app.post("/detect")
    def detect():
        try:
            result = pipeline(request.get_data())
        except ValueError:
            return {"error": "could not decode image"}, 400
        return jsonify(result)

    print(f"Serving on http://{HOST}:{PORT}", file=sys.stderr)
    app.run(host=HOST, port=PORT)


def run_call(image_path: Path) -> None:
    url = f"http://{HOST}:{PORT}/detect"
    with open(image_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n inference endpoint.")
    parser.add_argument(
        "--call",
        action="store_true",
        help="Send a test request to the running server and print the response.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Image to send with --call. Defaults to the sample COCO image.",
    )
    parser.add_argument("--assets-dir", type=Path, default=Path(".example_assets"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir

    if args.call:
        image_path = args.input or assets_dir / COCO_IMAGE_NAME
        if args.input is None:
            print(f"Downloading sample image to {image_path} if needed...", file=sys.stderr)
            download_if_missing(COCO_IMAGE_URL, image_path)
        run_call(image_path)
        return 0

    model_path = assets_dir / MODEL_NAME
    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)
    run_server(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
