from __future__ import annotations

import sys
from pathlib import Path

from examples.common import download_if_missing

YOLO8_MODELS: dict[str, tuple[str, str | None]] = {
    "n": ("yolov8n.onnx", "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"),
    "s": ("yolov8s.onnx", None),
    "m": ("yolov8m.onnx", None),
    "l": ("yolov8l.onnx", None),
    "x": ("yolov8x.onnx", None),
}


def resolve_model_variant_path(
    assets_dir: Path,
    model_name: str,
    model_url: str | None,
    variant: str,
) -> Path | None:
    """Return the model path, downloading it if a URL is provided."""
    model_path = assets_dir / model_name
    if model_url:
        download_if_missing(model_url, model_path)
    elif not model_path.exists():
        print(
            f"Model not found at {model_path}. "
            f"Export with: yolo export model=yolov8{variant}.pt format=onnx",
            file=sys.stderr,
        )
        return None
    return model_path
