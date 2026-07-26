from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from ml_pipes.core import Pipeline
from ml_pipes.standard import (
    Recall,
    Store,
)
from ml_pipes.vision import (
    Decode,
    Detections,
    DrawBoxes,
    DrawMasks,
    ImagePayload,
    LoadFile,
    SaveImage,
    Segmentations,
)

ASSETS_DIR = Path(__file__).resolve().parent / ".example_assets"

COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_IMAGE_NAME = "coco_000000039769.jpg"

SAMPLE_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
SAMPLE_VIDEO_NAME = "vtest.avi"

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def download_if_missing(url: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def decode() -> Pipeline[str | Path, ImagePayload]:
    return Pipeline([
        LoadFile(),
        Decode(),
        Store("source_image"),
    ])


def visualize_and_store(
    output_path: Path,
    class_names: list[str] | None = None,
) -> Pipeline[Segmentations, tuple[ImagePayload, Segmentations]]:
    return Pipeline([
        Recall("source_image", prepend=True),
        DrawMasks(class_names=class_names),
        DrawBoxes(class_names=class_names),
        SaveImage(output_path, at=0),
    ])


def visualize_detections_and_store(
    output_path: Path,
    class_names: list[str] | None = None,
) -> Pipeline[Detections, tuple[ImagePayload, Detections]]:
    return Pipeline([
        Recall("source_image", prepend=True),
        DrawBoxes(class_names=class_names),
        SaveImage(output_path, at=0),
    ])


def resolve_model_path(
    model_path: Path | None,
    default_path: Path,
    default_url: str | None = None,
) -> Path:
    resolved_model_path = model_path or default_path
    if model_path is None and default_url is not None:
        download_if_missing(default_url, resolved_model_path)
    if resolved_model_path.exists():
        return resolved_model_path

    print(f"Error: model file not found: {resolved_model_path}", file=sys.stderr)
    if model_path is None:
        if default_url is None:
            print(
                "The bundled default model is expected under the assets directory. "
                "Pass --model-path path/to/model-file to use a different local model.",
                file=sys.stderr,
            )
        else:
            print(
                "The default example model could not be found after the download attempt. "
                "Pass --model-path path/to/model-file to use a different local model.",
                file=sys.stderr,
            )
    raise SystemExit(1)


def resolve_input_path(
    input_path: Path | None,
    default_path: Path,
    default_url: str | None = None,
) -> Path:
    resolved_input_path = input_path or default_path
    if input_path is None and default_url is not None:
        download_if_missing(default_url, resolved_input_path)
    if resolved_input_path.exists():
        return resolved_input_path

    print(f"Error: input file not found: {resolved_input_path}", file=sys.stderr)
    raise SystemExit(1)


def build_output_path(
    output_dir: Path,
    image_name: str | Path,
    model_name: str | Path,
) -> Path:
    image_path = Path(image_name)
    model_path = Path(model_name)
    suffix = model_path.stem.replace(".", "_")
    return output_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
