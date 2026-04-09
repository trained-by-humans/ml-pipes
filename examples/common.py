from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np

from ml_pipes import DecodeOp, Detections, DrawBoxesOp, ImagePayload, Pipeline, SaveImageOp, Segmentations


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


def render_and_save_segmentations(
    image_path: Path,
    segmentations: Segmentations,
    output_path: Path,
    class_names: list[str] | None = None,
    alpha: float = 0.45,
) -> None:
    source_image = DecodeOp()(image_path)
    image = source_image.array.copy()
    for mask, class_id in zip(segmentations.masks, segmentations.classes, strict=True):
        color = np.asarray(_class_color(int(class_id)), dtype=np.float32)
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 2:
            raise ValueError(f"Expected 2D segmentation mask, got shape {mask_bool.shape}")
        if np.any(mask_bool):
            blended = (1.0 - alpha) * image[mask_bool].astype(np.float32) + alpha * color
            image[mask_bool] = blended.astype(np.uint8)

    detections = Detections(
        boxes=segmentations.boxes,
        scores=segmentations.scores,
        classes=segmentations.classes,
    )
    boxed = DrawBoxesOp(class_names=class_names)(
        detections,
        ImagePayload(array=image, color_space=source_image.color_space, layout=source_image.layout),
    )
    SaveImageOp(output_path)(boxed)


def _class_color(class_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color palette from class id.
    return (
        int((37 * class_id + 17) % 255),
        int((91 * class_id + 53) % 255),
        int((17 * class_id + 191) % 255),
    )


def build_output_path(
    assets_dir: Path,
    image_name: str | Path,
    model_name: str | Path,
) -> Path:
    image_path = Path(image_name)
    model_path = Path(model_name)
    suffix = model_path.stem.replace(".", "_")
    return assets_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
