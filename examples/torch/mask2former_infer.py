from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.common import COCO_IMAGE_NAME, build_output_path
from ml_pipes import Decode, ImagePayload, LoadFile, Pick, Pipeline, Store
from ml_pipes.torch.types import TorchTensorRegistry

MASK2FORMER_MODEL_IDS: dict[str, str] = {
    "panoptic": "facebook/mask2former-swin-tiny-coco-panoptic",
    "instance": "facebook/mask2former-swin-tiny-coco-instance",
}

COCO_PANOPTIC_THING_IDS = frozenset(range(80))


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "Mask2Former examples require transformers and safetensors. "
            "Install them in the project environment before running these scripts."
        ) from exc
    return AutoImageProcessor, Mask2FormerForUniversalSegmentation


@dataclass(frozen=True)
class LoadedMask2Former:
    task: str
    model_id: str
    processor: Any
    model: torch.nn.Module
    class_names: list[str]
    thing_class_ids: frozenset[int]

    @classmethod
    def load(cls, task: str, device: str) -> LoadedMask2Former:
        AutoImageProcessor, Mask2FormerForUniversalSegmentation = _require_transformers()
        model_id = MASK2FORMER_MODEL_IDS[task]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).eval().to(device)
        id2label = getattr(model.config, "id2label", None) or {}
        num_labels = getattr(model.config, "num_labels", 0)
        class_names = [id2label.get(index, str(index)) for index in range(num_labels)]
        thing_class_ids = COCO_PANOPTIC_THING_IDS if task == "panoptic" else frozenset(range(num_labels))
        return cls(
            task=task,
            model_id=model_id,
            processor=processor,
            model=model,
            class_names=class_names,
            thing_class_ids=thing_class_ids,
        )


class Mask2FormerInfer:
    def __init__(self, bundle: LoadedMask2Former, device: str) -> None:
        self.bundle = bundle
        self.device = device

    def __call__(self, image: ImagePayload) -> TorchTensorRegistry:
        if image.layout != "HWC":
            raise ValueError(f"Mask2FormerInfer expects HWC image layout, got {image.layout}")
        rgb = self._to_rgb_array(image)
        pixel_values = self.bundle.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            outputs = self.bundle.model(pixel_values=pixel_values)
        return TorchTensorRegistry(
            {
                "class_queries_logits": outputs.class_queries_logits[0],
                "masks_queries_logits": outputs.masks_queries_logits[0],
            }
        )

    @staticmethod
    def _to_rgb_array(image: ImagePayload) -> np.ndarray:
        if image.color_space == "BGR":
            return np.ascontiguousarray(image.array[:, :, ::-1])
        if image.color_space == "RGB":
            return np.ascontiguousarray(image.array)
        raise ValueError(f"Mask2FormerInfer expects BGR or RGB input, got {image.color_space}")


class SplitImageAndShape:
    def __call__(self, image: ImagePayload) -> tuple[ImagePayload, tuple[int, int]]:
        if image.layout != "HWC":
            raise ValueError(f"SplitImageAndShape expects HWC image layout, got {image.layout}")
        height, width = image.array.shape[:2]
        return image, (height, width)


def build_mask2former_preprocess_pipeline() -> Pipeline:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            Store("source_image"),
            SplitImageAndShape(),
            Store("image_shape", index=1),
            Pick(0),
        ]
    )


def resolve_task_list(task: str) -> list[str]:
    return ["panoptic", "instance"] if task == "both" else [task]


def resolve_output_path(
    requested_output: Path | None,
    assets_dir: Path,
    task: str,
    domain: str,
) -> Path:
    if requested_output is not None:
        stem = requested_output.stem
        return requested_output.with_name(f"{stem}_{task}_{domain}{requested_output.suffix}")
    return build_output_path(assets_dir, COCO_IMAGE_NAME, f"mask2former_{task}_{domain}.png")
