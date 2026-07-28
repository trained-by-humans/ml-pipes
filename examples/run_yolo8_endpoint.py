"""
Minimal YOLOv8 inference endpoint.

Requires `flask`.

Run from the repo root:
    python examples/run_yolo8_endpoint.py
    python examples/run_yolo8_endpoint.py --call
    python examples/run_yolo8_endpoint.py --call --input path/to/photo.jpg
"""
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
from run_yolo8_onnx import BUNDLED_MODEL_PATH, yolo8_inference_pipeline
from ml_pipes.core import (
    Embed,
    Pipeline,
)
from ml_pipes.vision import (
    Decode,
    MapPredictionsToObjects,
)

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
    parser.add_argument("--model-path", type=Path, default=None, help="Path to a local ONNX model. Defaults to the bundled YOLOv8n model in the assets directory.")
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

    model_path = resolve_model_path(args.model_path, BUNDLED_MODEL_PATH)
    run_server(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
