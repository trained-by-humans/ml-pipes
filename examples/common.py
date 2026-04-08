from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from ml_pipes import DecodeOp, DrawBoxesOp, Pipeline, SaveImageOp


COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_IMAGE_NAME = "coco_000000039769.jpg"

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


def render_and_save_detections(
    image_path: Path,
    detections: object,
    output_path: Path,
    class_names: list[str] | None = None,
) -> None:
    source_image = DecodeOp()(image_path)
    Pipeline(
        [
            lambda value: (value, source_image.array),
            DrawBoxesOp(class_names=class_names),
            SaveImageOp(output_path),
        ]
    )(detections)


def build_output_path(
    assets_dir: Path,
    image_name: str | Path,
    model_name: str | Path,
) -> Path:
    image_path = Path(image_name)
    model_path = Path(model_name)
    suffix = model_path.stem.replace(".", "_")
    return assets_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
