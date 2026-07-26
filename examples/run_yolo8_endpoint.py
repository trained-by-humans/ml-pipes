from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from common import (
    ASSETS_DIR,
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    resolve_input_path,
    resolve_model_path,
)
from run_yolo8_onnx import BUNDLED_MODEL_NAME, yolo8_inference_pipeline
from ml_pipes.core import (
    Embed,
    Pipeline,
)
from ml_pipes.vision import (
    Decode,
    MapPredictionsToObjects,
)

# Minimal YOLOv8 inference endpoint.
#
# Requires Flask:
#   pip install flask
#
# Commands below are shown from `examples/`. From the repository root, prefix
# script paths and asset paths with `examples/`.
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


def build_pipeline(model_path: Path) -> Pipeline[bytes, list[dict[str, object]]]:
    return Pipeline([
        Decode(),
        Embed(yolo8_inference_pipeline(model_path)),
        MapPredictionsToObjects(fields={"box": "boxes", "score": "scores", "class_id": "classes"}),
    ], auto_validate=True)


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
        "--model-path",
        type=Path,
        default=None,
        help="Path to a local ONNX model. Defaults to the bundled yolov8n model in the assets directory.",
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.call:
        image_path = resolve_input_path(args.input, ASSETS_DIR / COCO_IMAGE_NAME, COCO_IMAGE_URL)
        run_call(image_path)
        return 0

    model_path = resolve_model_path(args.model_path, ASSETS_DIR / BUNDLED_MODEL_NAME)
    run_server(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
