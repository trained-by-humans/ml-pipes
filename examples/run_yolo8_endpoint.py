from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    download_if_missing,
    resolve_model_path,
)
from run_yolo8_onnx import YOLO8_MODELS, yolo8_inference_pipeline
from ml_pipes import (
    Decode,
    Embed,
    MapToObjects,
    Pipeline,
)

# Minimal YOLOv8 inference endpoint.
#
# Requires Flask:
#   pip install flask
#
# Start the server:
#   python run_yolo8_endpoint.py
#
# Run a test call (downloads the sample COCO image if needed):
#   python run_yolo8_endpoint.py --call
#
# Or send a specific image:
#   python run_yolo8_endpoint.py --call --input photo.jpg
#
# Or with curl:
#   curl -s -X POST http://localhost:5000/detect \
#        -H "Content-Type: application/octet-stream" \
#        --data-binary @.example_assets/coco_000000039769.jpg | python -m json.tool

HOST = "localhost"
PORT = 5000


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline([
        Decode(),
        Embed(yolo8_inference_pipeline(model_path)),
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
    parser = argparse.ArgumentParser(description="YOLOv8 inference endpoint.")
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

    model_name, model_url = YOLO8_MODELS["n"]
    model_path = resolve_model_path(assets_dir, model_name, model_url, "n")
    if model_path is None:
        return 1
    run_server(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
