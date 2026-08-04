from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from examples.common import ASSETS_DIR, COCO_IMAGE_NAME, build_output_path
from ml_pipes.core import Pipeline
from ml_pipes.standard import Store
from ml_pipes.tensor import TensorPayload
from ml_pipes.torch import ToTorch, TorchExtract, TorchInfer, TorchSqueeze
from ml_pipes.torch.types import TorchTensorRegistry
from ml_pipes.vision import (
    ConvertColorSpace,
    Decode,
    ImagePayload,
    LoadFile,
)

MASK2FORMER_MODEL_IDS: dict[str, str] = {
    "panoptic": "facebook/mask2former-swin-tiny-coco-panoptic",
    "instance": "facebook/mask2former-swin-tiny-coco-instance",
}

COCO_PANOPTIC_THING_IDS = frozenset(range(80))


def add_mask2former_args(parser: argparse.ArgumentParser, *, device_help: str) -> None:
    parser.add_argument("--task", choices=("instance", "panoptic"), default="instance", help="Segmentation task to run (default: instance).")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help=device_help,
    )
    parser.add_argument("--input", type=Path, default=None, help="Input image path. Defaults to the sample COCO image.")
    parser.add_argument("--output", type=Path, default=None, help="Output path prefix for annotated images.")


def _require_transformers() -> tuple[Any, Any]:
    try:
        import safetensors  # noqa: F401
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError:
        print(
            "Transformers, safetensors, and scipy are required: "
            "python -m pip install transformers safetensors scipy",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return AutoImageProcessor, Mask2FormerForUniversalSegmentation


@dataclass(frozen=True)
class Mask2FormerBundle:
    task: str
    model_id: str
    processor: Any
    model: torch.nn.Module
    class_names: list[str]
    thing_class_ids: frozenset[int]

    @classmethod
    def load(cls, task: str, device: str) -> Mask2FormerBundle:
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


class PrepareHFImageInputs:
    def __init__(
        self,
        processor: Any,
        *,
        output_key: str,
        input_layout: str = "HWC",
        input_color_space: str = "RGB",
        input_channels: int | None = 3,
        output_layout: str = "NCHW",
        require_contiguous: bool = True,
    ) -> None:
        self.processor = processor
        self.output_key = output_key
        self.input_layout = input_layout
        self.input_color_space = input_color_space
        self.input_channels = input_channels
        self.output_layout = output_layout
        self.require_contiguous = require_contiguous

    def __call__(self, image: ImagePayload) -> TensorPayload:
        if image.layout != self.input_layout or image.color_space != self.input_color_space:
            raise ValueError(
                "PrepareHFImageInputs expects the configured ImagePayload contract, "
                f"got layout={image.layout!r} color_space={image.color_space!r}"
            )
        if image.array.ndim != len(self.input_layout):
            raise ValueError(
                "PrepareHFImageInputs expects an image matching the configured layout rank, "
                f"got shape {image.array.shape!r}"
            )
        if self.input_channels is not None:
            try:
                channel_axis = self.input_layout.index("C")
            except ValueError as exc:
                raise ValueError(
                    f"PrepareHFImageInputs input_layout={self.input_layout!r} must contain 'C' when input_channels is set"
                ) from exc
            if image.array.shape[channel_axis] != self.input_channels:
                raise ValueError(
                    "PrepareHFImageInputs expects the configured channel count, "
                    f"got shape {image.array.shape!r}"
                )
        if self.require_contiguous and not image.array.flags.c_contiguous:
            raise ValueError("PrepareHFImageInputs expects a contiguous image array")
        model_inputs = self.processor(images=image.array, return_tensors="np")
        output_array = np.asarray(model_inputs[self.output_key])
        return TensorPayload(
            array=output_array,
            layout=self.output_layout,
            dtype=str(output_array.dtype),
        )


def build_mask2former_infer_pipeline(
    bundle: Mask2FormerBundle,
    device: str,
) -> Pipeline[str | Path, TorchTensorRegistry]:
    return Pipeline(
        [
            LoadFile(),
            Decode(),
            Store("source_image"),
            Store("image_shape", source="spatial_shape"),
            ConvertColorSpace("RGB"),
            PrepareHFImageInputs(processor=bundle.processor, output_key="pixel_values"),
            ToTorch(device=device),
            TorchInfer(
                bundle.model,
                input_name="pixel_values",
                input_layout="NCHW",
            ),
            TorchExtract("class_queries_logits", "masks_queries_logits"),
            TorchSqueeze("class_queries_logits", axis=0),
            TorchSqueeze("masks_queries_logits", axis=0),
        ]
    )


def resolve_output_path(
    requested_output: Path | None,
    input_name: str | Path,
    task: str,
    domain: str,
) -> Path:
    if requested_output is not None:
        stem = requested_output.stem
        return requested_output.with_name(f"{stem}_{task}_{domain}{requested_output.suffix}")
    return build_output_path(ASSETS_DIR, input_name, f"mask2former_{task}_{domain}.png")
