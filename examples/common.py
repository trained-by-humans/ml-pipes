from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from ml_pipes import Decode, DrawBoxes, DrawMasks, LoadFile, Pipeline, Recall, SaveImage, Store

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


def decode() -> Pipeline:
    return Pipeline([
        LoadFile(),
        Decode(),
        Store("source_image"),
    ])


def visualize_and_store(output_path: Path, class_names: list[str] | None = None) -> Pipeline:
    return Pipeline([
        Recall("source_image", index=0),
        DrawMasks(class_names=class_names),
        DrawBoxes(class_names=class_names),
        SaveImage(output_path, at=0),
    ])


def visualize_detections_and_store(output_path: Path, class_names: list[str] | None = None) -> Pipeline:
    return Pipeline([
        Recall("source_image", index=0),
        DrawBoxes(class_names=class_names),
        SaveImage(output_path, at=0),
    ])


def build_output_path(
        assets_dir: Path,
        image_name: str | Path,
        model_name: str | Path,
) -> Path:
    image_path = Path(image_name)
    model_path = Path(model_name)
    suffix = model_path.stem.replace(".", "_")
    return assets_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
