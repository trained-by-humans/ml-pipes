from __future__ import annotations

from pathlib import Path

from examples.common import ASSETS_DIR, resolve_model_path

YOLO8_MODELS: dict[str, tuple[str, str | None]] = {
    "n": ("yolov8n.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx"),
    "s": ("yolov8s.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8s.onnx"),
    "m": ("yolov8m.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8m.onnx"),
    "l": ("yolov8l.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8l.onnx"),
    "x": ("yolov8x.onnx", "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8x.onnx"),
}


def resolve_model_variant(variant: str) -> tuple[str, Path]:
    """Return the benchmark model name and resolved local path for a YOLOv8 variant."""
    model_name, model_url = YOLO8_MODELS[variant]
    model_path = resolve_model_path(None, ASSETS_DIR / model_name, model_url)
    return model_name, model_path
