from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from ml_pipes.operator import Operator
from ml_pipes.standard import SideEffectOp
from ml_pipes.tensor import TensorPayload
from .types import ImagePayload, ResizeTransform

__all__ = [
    "ConvertColorSpace",
    "Decode",
    "ImagePayload",
    "LoadFile",
    "Normalize",
    "Resize",
    "ResizeTransform",
    "SaveImage",
]

PayloadT = TypeVar("PayloadT")


@Operator
class LoadFile:
    def __call__(self, image_path: str | Path) -> bytes:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()


@Operator
class Decode:
    def __call__(self, data: bytes) -> ImagePayload:
        import cv2

        image_bytes = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return ImagePayload(array=image, color_space="BGR", layout="HWC")


@Operator
class Resize:
    def __init__(
        self,
        target_size: tuple[int, int] = (640, 640),
        mode: Literal["letterbox", "resize"] = "letterbox",
        pad_value: int = 114,
        interpolation: Literal["nearest", "linear", "cubic", "area"] = "linear",
        center: bool = True,
        allow_scale_up: bool = True,
    ):
        self.size = target_size
        self.mode = mode
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.center = center
        self.allow_scale_up = allow_scale_up

    def __call__(self, image_payload: ImagePayload) -> tuple[ImagePayload, ResizeTransform]:
        import cv2

        self._validate_image_payload(image_payload)
        image = image_payload.array
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.size
        interpolation = self._resolve_interpolation(cv2)

        if self.mode == "letterbox":
            ratio = min(target_h / original_h, target_w / original_w)
            if not self.allow_scale_up:
                ratio = min(ratio, 1.0)

            resized_w = int(round(original_w * ratio))
            resized_h = int(round(original_h * ratio))
            resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

            dw = target_w - resized_w
            dh = target_h - resized_h
            if self.center:
                left = int(np.floor(dw / 2))
                right = int(np.ceil(dw / 2))
                top = int(np.floor(dh / 2))
                bottom = int(np.ceil(dh / 2))
            else:
                left = 0
                top = 0
                right = int(dw)
                bottom = int(dh)

            resized = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(self.pad_value, self.pad_value, self.pad_value),
            )
            scale = (ratio, ratio)
            pad = (float(left), float(top))
        elif self.mode == "resize":
            resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
            scale = (target_w / original_w, target_h / original_h)
            pad = (0.0, 0.0)
        else:
            raise ValueError(f"Unsupported resize mode: {self.mode}")

        transform = ResizeTransform(
            scale=scale,
            pad=pad,
            original_shape=(original_h, original_w),
            resized_shape=resized.shape[:2],
        )
        payload = ImagePayload(
            array=resized,
            color_space=image_payload.color_space,
            layout=image_payload.layout,
        )
        return payload, transform

    @staticmethod
    def _validate_image_payload(payload: ImagePayload) -> None:
        if payload.layout != "HWC":
            raise ValueError(f"Resize expects HWC image layout, got {payload.layout}")

    def _resolve_interpolation(self, cv2: object) -> int:
        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        return mapping[self.interpolation]


@Operator
class ConvertColorSpace:
    def __init__(self, output_color_space: Literal["RGB", "BGR"]):
        if output_color_space not in {"RGB", "BGR"}:
            raise ValueError(f"ConvertColorSpace only supports RGB or BGR output, got {output_color_space}")
        self.output_color_space = output_color_space

    def __call__(self, image_payload: ImagePayload) -> ImagePayload:
        if "C" not in image_payload.layout:
            raise ValueError(f"ConvertColorSpace expects a layout containing C, got {image_payload.layout}")

        if image_payload.color_space not in {"RGB", "BGR"}:
            raise ValueError(
                f"ConvertColorSpace only supports BGR/RGB input, got {image_payload.color_space}"
            )

        channel_axis = image_payload.layout.index("C")
        channels = image_payload.channels
        if channels != 3:
            raise ValueError(
                f"ConvertColorSpace only supports 3-channel images, got {channels} for layout {image_payload.layout}"
            )

        array = image_payload.array
        if image_payload.color_space != self.output_color_space:
            array = np.flip(array, axis=channel_axis)
        converted = np.ascontiguousarray(array)
        return ImagePayload(
            array=converted,
            color_space=self.output_color_space,
            layout=image_payload.layout,
        )


@Operator
class Normalize:
    def __init__(
        self,
        scale: float = 1.0 / 255.0,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        output_layout: Literal["NCHW", "NHWC", "CHW", "HWC"] = "NCHW",
        output_color_space: Literal["RGB", "BGR"] = "RGB",
        add_batch_dim: bool = True,
    ):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.output_layout = output_layout
        self.output_color_space = output_color_space
        self.add_batch_dim = add_batch_dim

    def __call__(self, image_payload: ImagePayload) -> TensorPayload:
        if image_payload.layout != "HWC":
            raise ValueError(f"Normalize expects HWC image layout, got {image_payload.layout}")

        image = image_payload.array
        if image_payload.color_space != self.output_color_space and {
            image_payload.color_space,
            self.output_color_space,
        } == {"BGR", "RGB"}:
            image = image[..., ::-1]
        elif image_payload.color_space != self.output_color_space:
            raise ValueError(
                f"Normalize cannot convert {image_payload.color_space} to {self.output_color_space}"
            )

        if np.issubdtype(image.dtype, np.floating):
            tensor = image.copy()
        else:
            tensor = image.astype(np.float32)
        tensor = tensor * self.scale
        if self.mean is not None:
            tensor = tensor - np.asarray(self.mean, dtype=tensor.dtype)
        if self.std is not None:
            tensor = tensor / np.asarray(self.std, dtype=tensor.dtype)

        if self.output_layout in {"NCHW", "CHW"}:
            tensor = np.transpose(tensor, (2, 0, 1))
        elif self.output_layout not in {"NHWC", "HWC"}:
            raise ValueError(f"Unsupported output layout: {self.output_layout}")

        final_layout = self.output_layout
        if self.add_batch_dim:
            tensor = np.expand_dims(tensor, axis=0)
            if self.output_layout in {"CHW", "HWC"}:
                final_layout = f"N{self.output_layout}"

        return TensorPayload(array=tensor, layout=final_layout, dtype=str(tensor.dtype))
@Operator
class SaveImage(SideEffectOp[PayloadT], Generic[PayloadT]):
    def __init__(self, output_path: str | Path, at: int | None = None):
        self.output_path = Path(output_path)
        self.at = at

    def effect(self, payload: PayloadT) -> None:
        import cv2

        payload_value: Any = payload
        image_payload = payload_value[self.at] if self.at is not None else payload_value
        if image_payload.layout != "HWC":
            raise ValueError(f"SaveImage expects HWC image layout, got {image_payload.layout}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(self.output_path), image_payload.array)
        if not written:
            raise ValueError(f"Failed to write image: {self.output_path}")
