from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, get_args, get_origin, overload

import numpy as np

from ml_pipes._typing.annotation import is_assignable
from ml_pipes.operator import Operator
from ml_pipes.standard import SideEffectOp
from ml_pipes.tensor import TensorPayload
from ml_pipes.validation import PipelineValidationError
from .types import (
    BoxPrediction,
    ClassPrediction,
    ImagePayload,
    Prediction,
    PredictionMask,
    ResizeTransform,
    ScorePrediction,
)

__all__ = [
    "ConvertColorSpace",
    "Decode",
    "FilterPredictions",
    "FilterPredictionsByArea",
    "FilterPredictionsByClass",
    "FilterPredictionsByScore",
    "ImagePayload",
    "LoadFile",
    "MapPredictionsToObjects",
    "Normalize",
    "Prediction",
    "Resize",
    "ResizeTransform",
    "SaveImage",
]

PredictionT = TypeVar("PredictionT", bound=Prediction)
ClassPredictionT = TypeVar("ClassPredictionT", bound=ClassPrediction)
ScorePredictionT = TypeVar("ScorePredictionT", bound=ScorePrediction)
BoxPredictionT = TypeVar("BoxPredictionT", bound=BoxPrediction)
PayloadT = TypeVar("PayloadT")
ObjectPrefixT = TypeVar("ObjectPrefixT")
ObjectIndexT = TypeVar("ObjectIndexT", bound=int | None)
ObjectMapping: TypeAlias = dict[str, object]


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
class FilterPredictions(Generic[PredictionT]):
    def __init__(self, predicate: Callable[[PredictionT], PredictionMask]):
        self.predicate = predicate

    def __call__(self, prediction: PredictionT) -> PredictionT:
        return prediction.filter(self.predicate(prediction))


@Operator
class FilterPredictionsByClass:
    def __init__(self, classes: Collection[int]):
        self.classes = frozenset(classes)

    def __call__(self, prediction: ClassPredictionT) -> ClassPredictionT:
        return prediction.filter([class_id in self.classes for class_id in prediction.classes])


@Operator
class FilterPredictionsByScore:
    def __init__(self, min_score: float):
        self.min_score = min_score

    def __call__(self, prediction: ScorePredictionT) -> ScorePredictionT:
        return prediction.filter([score >= self.min_score for score in prediction.scores])


@Operator
class FilterPredictionsByArea:
    def __init__(self, min_area: float = 0, max_area: float | None = None):
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, prediction: BoxPredictionT) -> BoxPredictionT:
        return prediction.filter([
            (x2 - x1) * (y2 - y1) >= self.min_area
            and (self.max_area is None or (x2 - x1) * (y2 - y1) <= self.max_area)
            for x1, y1, x2, y2 in prediction.boxes
        ])


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


@Operator
class MapPredictionsToObjects(Generic[ObjectIndexT, PredictionT]):
    @overload
    def __init__(
        self: "MapPredictionsToObjects[None, PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self: "MapPredictionsToObjects[Literal[1], PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: Literal[1],
    ) -> None:
        ...

    @overload
    def __init__(
        self: "MapPredictionsToObjects[int, PredictionT]",
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: int,
    ) -> None:
        ...

    def __init__(
        self,
        fields: Mapping[str, str | Callable[[PredictionT], Sequence[object]]],
        at: int | None = None,
    ) -> None:
        self.fields = fields
        self.at = at

    @overload
    def __call__(
        self: "MapPredictionsToObjects[None, PredictionT]",
        payload: PredictionT,
    ) -> list[ObjectMapping]:
        ...

    @overload
    def __call__(
        self: "MapPredictionsToObjects[Literal[1], PredictionT]",
        payload: tuple[ObjectPrefixT, PredictionT],
    ) -> tuple[ObjectPrefixT, list[ObjectMapping]]:
        ...

    @overload
    def __call__(self, payload: object) -> Any:
        ...

    def __call__(self, payload: object) -> Any:
        prediction_arrays = self._resolve_prediction_value(payload)
        columns: dict[str, Sequence[object]] = {}
        for field_name, source in self.fields.items():
            if isinstance(source, str):
                try:
                    column = getattr(prediction_arrays, source)
                except AttributeError as exc:
                    raise AttributeError(
                        f"MapPredictionsToObjects field {field_name!r} references missing attribute "
                        f"{source!r} on {type(prediction_arrays).__name__}"
                    ) from exc
            else:
                column = source(prediction_arrays)
            columns[field_name] = column

        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"MapPredictionsToObjects requires equal-length collections, got lengths {sorted(lengths)}"
            )

        records: list[dict[str, object]] = []
        field_names = tuple(columns.keys())
        rows = zip(*(columns[field_name] for field_name in field_names), strict=True)
        for row in rows:
            records.append(dict(zip(field_names, row, strict=True)))
        if self.at is not None:
            payload_tuple = cast(tuple[object, ...], payload)
            return payload_tuple[:self.at] + (records,) + payload_tuple[self.at + 1 :]
        return records

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations
        if self.at is None:
            if current_output is not Any and is_assignable(current_output, Prediction):
                return (current_output,), list[ObjectMapping]
            return (Any,), Any

        if current_output is Any:
            return (Any,), Any

        if get_origin(current_output) is tuple:
            parts = get_args(current_output)
        elif isinstance(current_output, tuple):
            parts = current_output
        else:
            return (Any,), Any

        normalized_index = self.at if self.at >= 0 else len(parts) + self.at
        if normalized_index < 0 or normalized_index >= len(parts):
            error_type = validation_error_type or PipelineValidationError
            raise error_type(
                f"MapPredictionsToObjects(at={self.at}) is out of bounds for "
                f"{current_output} (length {len(parts)})"
            )
        if not is_assignable(parts[normalized_index], Prediction):
            return (Any,), Any
        updated_parts = parts[:normalized_index] + (list[ObjectMapping],) + parts[normalized_index + 1 :]
        return (current_output,), updated_parts

    def _resolve_prediction_value(self, payload: object) -> PredictionT:
        if self.at is None:
            return cast(PredictionT, payload)
        if not isinstance(payload, tuple):
            raise TypeError(
                f"MapPredictionsToObjects(at={self.at}) requires a tuple payload, got {type(payload)!r}"
            )
        return cast(PredictionT, payload[self.at])
