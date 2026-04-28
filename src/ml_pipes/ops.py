from __future__ import annotations

import contextlib
import json
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Literal, Any, get_args, get_origin
from typing import TextIO

import numpy as np

from .batch import BatchGate
from .region import RegionCloser, RegionOpener
from .types import ResizeTransform
from .types import (
    Detections,
    ImagePayload,
    RuntimeOutputs,
    Segmentations,
    TensorPayload,
    TensorRegistry,
)


# ---------------------------------------------------------------------------
# Image / preprocessing
# ---------------------------------------------------------------------------

class LoadFile:
    def __call__(self, image_path: str | Path) -> bytes:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()


class Decode:
    def __call__(self, data: bytes) -> ImagePayload:
        import cv2

        image_bytes = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
        return ImagePayload(array=image, color_space="BGR", layout="HWC")


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

        payload = TensorPayload(array=tensor, layout=final_layout, dtype=str(tensor.dtype))
        return payload


class Cast:
    def __init__(self, dtype: str, field: str | None = None):
        self.dtype = np.dtype(dtype)
        self.field = field

    def resolve_contract(self, current_output, stored_annotations, expand_output_annotation, error_type):
        return (Any,), current_output  # cast changes precision, not the payload type

    def __call__(self, value: object) -> object:
        if self.field is not None:
            selected = getattr(value, self.field)
            casted = self._cast_tensor_value(selected)
            if is_dataclass(value):
                return replace(value, **{self.field: casted})
            raise TypeError(
                f"Cast field={self.field!r} requires a dataclass value, got {type(value)!r}"
            )
        return self._cast_tensor_value(value)

    def _cast_tensor_value(self, value: object) -> TensorPayload | tuple[TensorPayload, ...]:
        if isinstance(value, TensorPayload):
            return TensorPayload(array=value.array.astype(self.dtype), layout=value.layout, dtype=str(self.dtype))
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return tuple(
                TensorPayload(array=tensor.array.astype(self.dtype), layout=tensor.layout, dtype=str(self.dtype))
                for tensor in value
            )
        raise TypeError(f"Cast does not support value type {type(value)!r}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class Infer:
    def __init__(
        self,
        model_path: str | Path,
        # Runtime-facing execution provider config.
        providers: tuple[str, ...] = ("CoreMLExecutionProvider", "CPUExecutionProvider"),
        # Runtime-facing input binding. Input names come from the exported graph.
        input_name: str | None = None,
        # Runtime-facing input tensor contract.
        input_layout: str = "NCHW",
        # Model-facing input dtype contract.
        dtype: str | None = None,
        # Runtime-facing output tensor metadata aligned with exported graph output order.
        output_layouts: tuple[str, ...] | None = None,
        # Serialize session.run() calls with a lock.  Off by default because
        # runtimes like ONNX Runtime manage their own internal thread pool and
        # handle concurrent calls efficiently.  Enable if profiling shows that
        # concurrent calls are oversubscribing CPU cores on your hardware.
        serialize: bool = False,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

        import onnxruntime as ort

        self.model_path = path
        self.session = ort.InferenceSession(
            str(path),
            providers=list(providers),
        )
        self._lock = threading.Lock() if serialize else contextlib.nullcontext()
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.input_layout = input_layout
        self.model_dtype = np.dtype(dtype) if dtype is not None else None
        self.output_layouts = output_layouts
        self.output_names = tuple(output.name for output in self.session.get_outputs())

    def __call__(self, tensor_payload: TensorPayload) -> RuntimeOutputs:
        if tensor_payload.layout != self.input_layout:
            raise ValueError(
                f"Infer expects {self.input_layout} tensor layout, got {tensor_payload.layout}"
            )

        actual_dtype = np.dtype(tensor_payload.dtype)
        if self.model_dtype is not None and actual_dtype != self.model_dtype:
            raise ValueError(f"Infer expects model dtype {self.model_dtype}, got {actual_dtype}")

        with self._lock:
            outputs = self.session.run(None, {self.input_name: tensor_payload.array})

        if self.output_layouts is None:
            output_layouts = tuple("UNKNOWN" for _ in outputs)
        else:
            if len(self.output_layouts) != len(outputs):
                raise ValueError(
                    f"Infer expected {len(self.output_layouts)} output layouts, got {len(outputs)} outputs"
                )
            output_layouts = self.output_layouts

        runtime_output_names = self.output_names or tuple(f"output_{index}" for index in range(len(outputs)))
        tensors = tuple(
            TensorPayload(array=np.asarray(output), layout=layout, dtype=str(np.asarray(output).dtype))
            for output, layout in zip(outputs, output_layouts, strict=True)
        )
        return RuntimeOutputs(tensors=tensors, names=runtime_output_names)


# ---------------------------------------------------------------------------
# Registry creation
# ---------------------------------------------------------------------------

class Extract:
    """Extracts named tensors from RuntimeOutputs into a TensorRegistry.

    Single output with rename:  Extract("output0", as_="preds")
    Multiple outputs:           Extract("output0", "output1", as_=("preds", "protos"))
    """

    def __init__(self, *names: str, as_: str | tuple[str, ...] | None = None):
        if not names:
            raise ValueError("Extract requires at least one output name")
        if as_ is not None:
            aliases: tuple[str, ...] = (as_,) if isinstance(as_, str) else tuple(as_)
            if len(aliases) != len(names):
                raise ValueError(
                    f"Extract: as_ length ({len(aliases)}) must match names length ({len(names)})"
                )
        else:
            aliases = names
        self._mapping: dict[str, str] = dict(zip(names, aliases))

    def __call__(self, outputs: RuntimeOutputs) -> TensorRegistry:
        registry = TensorRegistry()
        for src, dst in self._mapping.items():
            if src not in outputs.names:
                raise KeyError(
                    f"Extract: output {src!r} not found. Available: {list(outputs.names)}"
                )
            idx = list(outputs.names).index(src)
            registry[dst] = outputs.tensors[idx].array
        return registry


# ---------------------------------------------------------------------------
# Tensor shape manipulation
# ---------------------------------------------------------------------------

class Squeeze:
    """Removes size-1 dimensions from a named tensor."""

    def __init__(self, src: str, axis: int | tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        tensor = registry[self.src]
        registry[self.as_] = np.squeeze(tensor, axis=self.axis) if self.axis is not None else np.squeeze(tensor)
        return registry


class Transpose:
    """Transposes a named tensor."""

    def __init__(self, src: str, axes: tuple[int, ...] | None = None, as_: str | None = None):
        self.src = src
        self.axes = axes
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = np.transpose(registry[self.src], self.axes)
        return registry


# ---------------------------------------------------------------------------
# Tensor indexing
# ---------------------------------------------------------------------------

class Slice:
    """Slices columns from a 2D named tensor: as_ = src[:, s].

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, at: slice, as_: str | None = None):
        self.src = src
        self.at = at
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = registry[self.src][:, self.at]
        return registry


class GatherRows:
    """Gathers values: as_ = src[arange(N), indices].

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, indices: str, as_: str | None = None):
        self.src = src
        self.indices = indices
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        src = registry[self.src]
        idx = registry[self.indices]
        registry[self.as_] = src[np.arange(src.shape[0]), idx]
        return registry


# ---------------------------------------------------------------------------
# Math / activations
# ---------------------------------------------------------------------------

class ArgMax:
    """Computes argmax along an axis: as_ = argmax(src, axis).

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = np.argmax(registry[self.src], axis=self.axis).astype(np.int32)
        return registry


class GatherScores:
    """Reduces a 2D score matrix to 1D by picking each row's value at its class index.

    Equivalent to: scores[arange(N), classes]
    Writes result back to scores (or as_) in the registry.
    """

    def __init__(self, scores: str, classes: str, as_: str | None = None):
        self.scores = scores
        self.classes = classes
        self.as_ = as_ or scores

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        scores = registry[self.scores]
        classes = registry[self.classes]
        registry[self.as_] = scores[np.arange(scores.shape[0]), classes].astype(scores.dtype)
        return registry


class Softmax:
    """Applies softmax along an axis.

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, axis: int = -1, as_: str | None = None):
        self.src = src
        self.axis = axis
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        x = registry[self.src]
        shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp = np.exp(shifted)
        registry[self.as_] = exp / np.sum(exp, axis=self.axis, keepdims=True)
        return registry


class Sigmoid:
    """Applies sigmoid elementwise.

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, as_: str | None = None):
        self.src = src
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = 1.0 / (1.0 + np.exp(-registry[self.src]))
        return registry


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Scale:
    """Multiplies a tensor element-wise by a scalar or a per-column broadcast array.

    Useful for converting normalized coordinates to pixel space (denormalize)
    or pixel coordinates to [0, 1] space (normalize).
    Defaults to in-place (overwrites src) when as_ is not provided.

    Examples:
      Scale("boxes", by=640.0)                           # uniform scale
      Scale("boxes", by=(width, height, width, height))  # per-column for cxcywh / xyxy
    """

    def __init__(self, src: str, by: float | tuple | list, as_: str | None = None):
        self.src = src
        self.by = np.asarray(by)
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        tensor = registry[self.src]
        registry[self.as_] = tensor * self.by.astype(tensor.dtype)
        return registry


BoxFormat = Literal["xyxy", "xywh", "cxcywh"]
_BOX_FORMATS: frozenset[str] = frozenset(get_args(BoxFormat))


class ConvertBoxFormat:
    """Converts bounding boxes between coordinate formats.

    Supported formats:
      "xyxy"   — (x1, y1, x2, y2) corner coordinates
      "xywh"   — (x, y, w, h) top-left corner + size
      "cxcywh" — (cx, cy, w, h) center + size  (YOLO model output)

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str = "boxes", *, from_: BoxFormat, to: BoxFormat = "xyxy", as_: str | None = None):
        if from_ not in _BOX_FORMATS:
            raise ValueError(f"ConvertBoxFormat: unknown from_ format {from_!r}. Choose from {sorted(_BOX_FORMATS)}")
        if to not in _BOX_FORMATS:
            raise ValueError(f"ConvertBoxFormat: unknown to format {to!r}. Choose from {sorted(_BOX_FORMATS)}")
        self.src = src
        self.from_ = from_
        self.to = to
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        boxes = registry[self.src]
        registry[self.as_] = self._convert(boxes, self.from_, self.to)
        return registry

    @staticmethod
    def _convert(boxes: np.ndarray, from_: str, to: str) -> np.ndarray:
        if from_ == to:
            return boxes

        # Normalise everything to xyxy first, then convert to target
        if from_ == "xyxy":
            xyxy = boxes
        elif from_ == "xywh":
            xyxy = np.concatenate(
                [boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], axis=1
            )
        elif from_ == "cxcywh":
            half = boxes[:, 2:4] / 2.0
            xyxy = np.concatenate([boxes[:, :2] - half, boxes[:, :2] + half], axis=1)
        else:
            raise ValueError(from_)

        if to == "xyxy":
            return xyxy.astype(boxes.dtype)
        if to == "xywh":
            return np.concatenate(
                [xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2]], axis=1
            ).astype(boxes.dtype)
        if to == "cxcywh":
            wh = xyxy[:, 2:4] - xyxy[:, :2]
            return np.concatenate(
                [xyxy[:, :2] + wh / 2.0, wh], axis=1
            ).astype(boxes.dtype)
        raise ValueError(to)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class NMS:
    """Non-Maximum Suppression on named tensors in a TensorRegistry.

    Filters boxes, scores, and classes in-place.
    Optionally stores the kept indices under kept_as for use with FilterBy.
    """

    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        kept_as: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.kept_as = kept_as
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        boxes = registry[self.boxes]
        scores = registry[self.scores]
        classes = registry[self.classes]

        conf_mask = scores >= self.conf_threshold
        filtered_boxes = boxes[conf_mask]
        filtered_scores = scores[conf_mask]
        filtered_classes = classes[conf_mask]
        original_indices = np.where(conf_mask)[0]

        if filtered_boxes.size == 0:
            kept_original = np.zeros((0,), dtype=np.int32)
        else:
            kept_filtered = self._nms_indices(filtered_boxes, filtered_scores, filtered_classes)
            kept_original = original_indices[kept_filtered]

        registry[self.boxes] = boxes[kept_original]
        registry[self.scores] = scores[kept_original]
        registry[self.classes] = classes[kept_original]
        if self.kept_as is not None:
            registry[self.kept_as] = kept_original
        return registry

    def _nms_indices(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        kept_indices: list[int] = []
        for class_id in np.unique(classes):
            class_indices = np.where(classes == class_id)[0]
            ordered = class_indices[np.argsort(scores[class_indices])[::-1]]

            while ordered.size > 0:
                current = int(ordered[0])
                kept_indices.append(current)
                if len(kept_indices) >= self.max_detections or ordered.size == 1:
                    break
                remaining = ordered[1:]
                ious = self._compute_iou(boxes[current], boxes[remaining])
                ordered = remaining[ious < self.iou_threshold]

            if len(kept_indices) >= self.max_detections:
                break

        if not kept_indices:
            return np.zeros((0,), dtype=np.int32)

        kept = np.asarray(kept_indices, dtype=np.int32)
        final_order = np.argsort(scores[kept])[::-1]
        return kept[final_order][: self.max_detections]

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
        intersection = inter_w * inter_h

        box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
        boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
            boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
        )
        union = np.clip(box_area + boxes_area - intersection, a_min=1e-9, a_max=None)
        return intersection / union


class NMM:
    """Non-Maximum Merge on a Detections object.

    Like NMS, boxes that overlap above *iou_threshold* are grouped together.
    Unlike NMS, the surviving box is the score-weighted average of all boxes
    in the group rather than the highest-score box unchanged.  This produces
    a more accurate centroid when the same object is detected from multiple
    overlapping tiles.

    Operates on ``Detections`` (boxes/scores/classes lists), not TensorRegistry.
    Pair with ``Stitch()`` for tiled inference::

        Stitch(),
        NMM(iou_threshold=0.5),
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def __call__(self, detections: "Detections") -> "Detections":
        from .types import Detections

        if not detections.boxes:
            return detections

        boxes = np.array(detections.boxes, dtype=np.float32)
        scores = np.array(detections.scores, dtype=np.float32)
        classes = np.array(detections.classes, dtype=np.int32)

        merged_boxes: list[list[float]] = []
        merged_scores: list[float] = []
        merged_classes: list[int] = []

        for class_id in np.unique(classes):
            idx = np.where(classes == class_id)[0]
            ordered = idx[np.argsort(scores[idx])[::-1]]
            consumed = np.zeros(len(ordered), dtype=bool)

            for i, current in enumerate(ordered):
                if consumed[i]:
                    continue
                remaining_mask = ~consumed
                remaining_mask[i] = False
                remaining = ordered[remaining_mask]
                if remaining.size > 0:
                    ious = NMS._compute_iou(boxes[current], boxes[remaining])
                    overlap_pos = np.where(ious >= self.iou_threshold)[0]
                    group = np.array([current, *remaining[overlap_pos]])
                    for pos in overlap_pos:
                        consumed[np.where(ordered == remaining[pos])[0]] = True
                else:
                    group = np.array([current])

                group_boxes = boxes[group]
                group_scores = scores[group]
                weights = group_scores / group_scores.sum()
                merged_box = (group_boxes * weights[:, None]).sum(axis=0).tolist()
                merged_boxes.append(merged_box)
                merged_scores.append(float(group_scores[0]))  # highest score
                merged_classes.append(int(class_id))

        return Detections(boxes=merged_boxes, scores=merged_scores, classes=merged_classes)


class FilterBy:
    """Filters a tensor by an index array stored in the registry: as_ = src[indices].

    Pair with NMS(kept_as=...) to synchronise extra tensors (e.g. mask coefficients)
    with the boxes/scores/classes that NMS already filtered.

    Defaults to in-place (overwrites src) when as_ is not provided.
    """

    def __init__(self, src: str, indices: str, as_: str | None = None):
        self.src = src
        self.indices = indices
        self.as_ = as_ or src

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        registry[self.as_] = registry[self.src][registry[self.indices]]
        return registry


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class ReconstructMasks:
    """Reconstructs raw segmentation masks from coefficients and prototypes.

    dst = (coefficients @ prototypes.reshape(C, -1)).reshape(N, H, W)
    """

    def __init__(self, coefficients: str, prototypes: str, as_: str):
        self.coefficients = coefficients
        self.prototypes = prototypes
        self.as_ = as_

    def __call__(self, registry: TensorRegistry) -> TensorRegistry:
        coefficients = registry[self.coefficients]  # (N, C)
        prototypes = registry[self.prototypes]       # (C, H, W)
        channels, mask_h, mask_w = prototypes.shape
        masks = coefficients @ prototypes.reshape(channels, -1)
        registry[self.as_] = masks.reshape(-1, mask_h, mask_w)
        return registry


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class ProjectBoxes:
    """Projects boxes from model space to original image space.

    Accepts (TensorRegistry, ResizeTransform) — use Recall to provide the transform.
    """

    def __init__(self, src: str = "boxes"):
        self.src = src

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        boxes = registry[self.src].copy()
        pad_x, pad_y = transform.pad
        scale_x, scale_y = transform.scale
        original_h, original_w = transform.original_shape

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))
        registry[self.src] = boxes
        return registry


class ProjectMasks:
    """Zeros prototype masks outside each bounding box, then upsamples to original image size.

    Zeroing is applied in prototype space (small tensor, vectorised across all N masks) before
    upsampling, which keeps the operation GPU-friendly — the expensive resize only runs on
    already-sparse data. Boxes are converted from original image space to prototype space
    internally, so this operator must be called AFTER ProjectBoxes.

    Accepts (TensorRegistry, ResizeTransform) — use Recall to provide the transform.
    """

    def __init__(self, masks: str = "masks", boxes: str = "boxes", mask_threshold: float = 0.5):
        self.masks = masks
        self.boxes = boxes
        self.mask_threshold = mask_threshold

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        import cv2

        masks = registry[self.masks]   # (N, proto_H, proto_W)
        boxes = registry[self.boxes]   # (N, 4) xyxy — original image space
        resized_h, resized_w = transform.resized_shape
        orig_h, orig_w = transform.original_shape
        scale_x, scale_y = transform.scale
        pad_x, pad_y = transform.pad
        _, proto_h, proto_w = masks.shape

        # Convert boxes from original image space to prototype space:
        #   original → model input:  x_model = x_orig * scale + pad
        #   model input → prototype: x_proto = x_model * (proto / resized)
        proto_boxes = boxes.astype(np.float32).copy()
        proto_boxes[:, [0, 2]] = (proto_boxes[:, [0, 2]] * scale_x + pad_x) * (proto_w / resized_w)
        proto_boxes[:, [1, 3]] = (proto_boxes[:, [1, 3]] * scale_y + pad_y) * (proto_h / resized_h)

        # Zero outside each box — vectorised on the small (N, proto_H, proto_W) tensor
        x1 = proto_boxes[:, 0].clip(0, proto_w)[:, None, None]
        y1 = proto_boxes[:, 1].clip(0, proto_h)[:, None, None]
        x2 = proto_boxes[:, 2].clip(0, proto_w)[:, None, None]
        y2 = proto_boxes[:, 3].clip(0, proto_h)[:, None, None]
        cols = np.arange(proto_w, dtype=np.float32)[None, None, :]
        rows = np.arange(proto_h, dtype=np.float32)[None, :, None]
        masks = masks * ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2))

        # Upsample each zeroed mask to original image size
        top    = max(int(round(pad_y - 0.1)), 0)
        left   = max(int(round(pad_x - 0.1)), 0)
        bottom = min(int(round(resized_h - pad_y + 0.1)), resized_h)
        right  = min(int(round(resized_w - pad_x + 0.1)), resized_w)

        projected = []
        for mask in masks:
            upsampled = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            projected.append((cv2.resize(upsampled[top:bottom, left:right], (orig_w, orig_h),
                                         interpolation=cv2.INTER_LINEAR) > self.mask_threshold).astype(np.uint8))

        registry[self.masks] = projected
        return registry


class ProjectRoIMasks:
    """Resizes per-instance RoI masks to their bounding boxes and embeds them into a full-image canvas.

    For models that output one small fixed-size mask per detection relative to its bounding box
    (e.g. Mask R-CNN), rather than a shared prototype feature map (cf. ProjectMasks).

    Expects masks of shape (N, H, W). If the model outputs (N, 1, H, W), add
    Squeeze("masks", axis=1) before this operator.

    Accepts (TensorRegistry, ResizeTransform) — use Recall to provide the transform.
    Must be called AFTER ProjectBoxes — needs boxes already in original image space.
    """

    def __init__(self, masks: str = "masks", boxes: str = "boxes", mask_threshold: float = 0.5):
        self.masks = masks
        self.boxes = boxes
        self.mask_threshold = mask_threshold

    def __call__(self, registry: TensorRegistry, transform: ResizeTransform) -> TensorRegistry:
        import cv2

        boxes = registry[self.boxes]   # (N, 4) xyxy — original image space
        masks = registry[self.masks]   # (N, H, W)
        orig_h, orig_w = transform.original_shape

        canvas = np.zeros((len(boxes), orig_h, orig_w), dtype=bool)
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            x1 = max(0, int(np.floor(box[0])))
            y1 = max(0, int(np.floor(box[1])))
            x2 = min(orig_w, int(np.ceil(box[2])))
            y2 = min(orig_h, int(np.ceil(box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            resized = cv2.resize(mask.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            canvas[i, y1:y2, x1:x2] = resized > self.mask_threshold

        registry[self.masks] = canvas
        return registry


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------

class ToDetections:
    """Converts named tensors in a TensorRegistry to a Detections output."""

    def __init__(self, boxes: str = "boxes", scores: str = "scores", classes: str = "classes"):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    def __call__(self, registry: TensorRegistry) -> Detections:
        return Detections(
            boxes=registry[self.boxes].tolist(),
            scores=registry[self.scores].tolist(),
            classes=registry[self.classes].tolist(),
        )


class ToSegmentations:
    """Converts named tensors in a TensorRegistry to a Segmentations output."""

    def __init__(
        self,
        boxes: str = "boxes",
        scores: str = "scores",
        classes: str = "classes",
        masks: str = "masks",
    ):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.masks = masks

    def __call__(self, registry: TensorRegistry) -> Segmentations:
        masks_data = registry[self.masks]
        return Segmentations(
            boxes=registry[self.boxes].tolist(),
            scores=registry[self.scores].tolist(),
            classes=registry[self.classes].tolist(),
            masks=list(masks_data) if isinstance(masks_data, np.ndarray) else masks_data,
        )


# ---------------------------------------------------------------------------
# Side-effect base class
# ---------------------------------------------------------------------------

class SideEffectOp(ABC):
    """Base for operators that perform a side effect and return their input unchanged.

    Subclasses implement `effect(payload)` instead of `__call__`. The base class
    owns `__call__` to enforce the passthrough contract — the input is always
    returned verbatim. `resolve_contract` threads the upstream type through so
    these operators work transparently in strict pipelines.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override __call__; implement effect() instead"
            )

    @abstractmethod
    def effect(self, payload: Any) -> None:
        raise NotImplementedError

    def __call__(self, payload: Any) -> Any:
        self.effect(payload)
        return payload

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        return (Any,), current_output


# ---------------------------------------------------------------------------
# Visualization / side-effects
# ---------------------------------------------------------------------------

class DrawBoxes:
    def __init__(
        self,
        class_names: list[str] | tuple[str, ...] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.class_names = tuple(class_names) if class_names is not None else None
        self.color = color
        self.thickness = thickness
        self.font_scale = font_scale

    def __call__(self, source_image: ImagePayload, detections: Detections) -> tuple[ImagePayload, Detections]:
        import cv2

        if source_image is None:
            raise ValueError("source_image missing from context; cannot draw detections")

        image = source_image.array.copy()

        for box, score, class_id in zip(detections.boxes, detections.scores, detections.classes, strict=True):
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, self.thickness)

            label = self._format_label(int(class_id), float(score))
            text_origin_y = y1 - 8 if y1 > 18 else y1 + 18
            cv2.putText(
                image,
                label,
                (x1, text_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                max(1, self.thickness - 1),
                cv2.LINE_AA,
            )

        return ImagePayload(array=image, color_space="BGR", layout="HWC"), detections

    def _format_label(self, class_id: int, score: float) -> str:
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = str(class_id)
        return f"{name} {score:.2f}"


class DrawMasks:
    def __init__(
        self,
        class_names: list[str] | tuple[str, ...] | None = None,
        alpha: float = 0.45,
    ):
        self.class_names = tuple(class_names) if class_names is not None else None
        self.alpha = alpha

    def __call__(self, source_image: ImagePayload, segmentations: Segmentations) -> tuple[ImagePayload, Segmentations]:
        if source_image is None:
            raise ValueError("source_image missing from context; cannot draw masks")

        image = source_image.array.copy()
        for mask, class_id in zip(segmentations.masks, segmentations.classes, strict=True):
            color = np.asarray(self._class_color(int(class_id)), dtype=np.float32)
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.ndim != 2:
                raise ValueError(f"Expected 2D segmentation mask, got shape {mask_bool.shape}")
            if np.any(mask_bool):
                blended = (1.0 - self.alpha) * image[mask_bool].astype(np.float32) + self.alpha * color
                image[mask_bool] = blended.astype(np.uint8)

        return ImagePayload(array=image, color_space=source_image.color_space, layout=source_image.layout), segmentations

    def _class_color(self, class_id: int) -> tuple[int, int, int]:
        return (
            int((37 * class_id + 17) % 255),
            int((91 * class_id + 53) % 255),
            int((17 * class_id + 191) % 255),
        )


class SaveImage(SideEffectOp):
    def __init__(self, output_path: str | Path, at: int | None = None):
        self.output_path = Path(output_path)
        self.at = at

    def effect(self, payload: Any) -> None:
        import cv2

        image_payload = payload[self.at] if self.at is not None else payload
        if image_payload.layout != "HWC":
            raise ValueError(f"SaveImage expects HWC image layout, got {image_payload.layout}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(self.output_path), image_payload.array)
        if not written:
            raise ValueError(f"Failed to write image: {self.output_path}")


class MapToObjects:
    # Intentionally uses `object → Any`: output type depends on runtime `at` indexing
    # and cannot be resolved without custom resolve_contract. Not a SideEffectOp.
    def __init__(
        self,
        fields: dict[str, str | Callable[[object], Sequence[object]]],
        at: int | None = None,
    ):
        self.fields = fields
        self.at = at

    def __call__(self, payload: object) -> Any:
        prediction_arrays = payload[self.at] if self.at is not None else payload
        columns: dict[str, Sequence[object]] = {}
        for field_name, source in self.fields.items():
            if isinstance(source, str):
                column = getattr(prediction_arrays, source)
            else:
                column = source(prediction_arrays)
            columns[field_name] = column

        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"MapToObjects requires equal-length collections, got lengths {sorted(lengths)}")

        records: list[dict[str, object]] = []
        field_names = tuple(columns.keys())
        rows = zip(*(columns[field_name] for field_name in field_names), strict=True)
        for row in rows:
            record = dict(zip(field_names, row, strict=True))
            records.append(record)
        if self.at is not None:
            return payload[:self.at] + (records,) + payload[self.at + 1:]
        return records


class LogDetections(SideEffectOp):
    def __init__(
        self,
        model_path: str | Path,
        image_path: str | Path,
        annotated_image_path: str | Path,
        indent: int = 2,
        stream: TextIO | None = None,
        at: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.image_path = Path(image_path)
        self.annotated_image_path = Path(annotated_image_path)
        self.indent = indent
        self.stream = stream or sys.stdout
        self.at = at

    def effect(self, payload: Any) -> None:
        prediction_objects = payload[self.at] if self.at is not None else payload
        print(
            json.dumps(
                {
                    "model": str(self.model_path),
                    "image": str(self.image_path),
                    "annotated_image": str(self.annotated_image_path),
                    "detections": prediction_objects,
                },
                indent=self.indent,
            ),
            file=self.stream,
        )


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

class Pick:
    """Selects one or more elements from a tuple by index, discarding the rest.

    A pure routing operator: it changes which value flows forward but never
    reads or writes the context. Commonly used after Store to discard the
    ResizeTransform and keep only the ImagePayload before inference.
    """

    def __init__(self, *indices: int):
        if not indices:
            raise ValueError("Pick requires at least one index")
        self.indices = indices

    def __call__(self, current: tuple) -> Any:
        if not isinstance(current, tuple):
            raise TypeError("Pick can only be applied to tuple outputs")
        selected = tuple(current[index] for index in self.indices)
        if len(selected) == 1:
            return selected[0]
        return selected

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        if current_output is None or get_origin(current_output) is not tuple:
            return (Any,), Any
        parts = get_args(current_output)
        selected = []
        for i in self.indices:
            if i >= len(parts):
                raise validation_error_type(
                    f"Pick({i}) is out of bounds for {current_output} (length {len(parts)})"
                )
            selected.append(parts[i])
        selected = tuple(selected)
        return (Any,), selected[0] if len(selected) == 1 else selected


# ---------------------------------------------------------------------------
# Batch coordination
# ---------------------------------------------------------------------------

class UnBatch(RegionCloser):
    """
    Batch coordination exit point.

    Stateless marker.  Pipeline detects this operator, calls
    ``gate.distribute()`` on the matching Batch's gate, and routes each
    thread's individual result to the remaining operators.
    """

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        # UnBatch unwraps list[T] back to T for the per-sample operators that follow.
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (Any,), args[0] if args else Any
        return (Any,), Any


class Batch(RegionOpener):
    """
    Batch coordination entry point.

    Multiple threads calling the same pipeline instance block here until
    *size* samples have arrived or *timeout* seconds have elapsed since the
    first arrival.  One thread is elected leader and continues through the
    batch region; the rest wait for their individual results.

    Pipeline handles the leader/waiter split; this operator owns the gate.

    Example::

        pipeline = Pipeline([
            ...,
            Batch(size=4, timeout=0.05),
            Collate(),
            Infer("model.onnx"),
            Distribute(),
            UnBatch(),
            ...,
        ])
    """

    closing_type = UnBatch

    def __init__(self, size: int, timeout: float = 0.05) -> None:
        self.gate = BatchGate(size, timeout)

    def run_region(
        self,
        current: Any,
        label: str,
        execute_region: Callable,
        trace: Any,
        cfg: Any,
    ) -> Any:
        import time
        from .batch import LeaderBatch
        from .tracing import InvocationTrace, StepSpan
        from .core import _NoOpTrace

        gate = self.gate

        t_gate_enter = time.perf_counter()
        outcome = gate.enter(current)
        gate_blocked_duration = time.perf_counter() - t_gate_enter

        if not isinstance(outcome, LeaderBatch):
            batch_region_duration = outcome.batch_span.duration_s if outcome.batch_span is not None else 0.0
            lobby_wait_duration = gate_blocked_duration - batch_region_duration
            trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, lobby_wait_duration))
            if outcome.batch_span is not None:
                trace.spans.append(outcome.batch_span)
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.result

        trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, gate_blocked_duration))
        current = outcome.inputs
        batch_size = len(current) if hasattr(current, "__len__") else None
        collecting = isinstance(trace, InvocationTrace)
        child_trace = InvocationTrace(batch_size=batch_size) if collecting else _NoOpTrace(batch_size=batch_size)

        t_region = time.perf_counter()
        try:
            current, child_trace = execute_region(current, child_trace)
        except Exception as exc:
            error_span = StepSpan(label, t_region, child_trace.total_duration_s, error=True, child_trace=child_trace if collecting else None)
            trace.spans.append(error_span)
            gate.distribute_exception(exc, batch_span=error_span if collecting else None)
            raise

        batch_span = StepSpan(label, t_region, child_trace.total_duration_s, child_trace=child_trace if collecting else None)
        trace.spans.append(batch_span)
        return gate.distribute(current, batch_span=batch_span if collecting else None)

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        # Batch collects individual samples into a list for the batch region.
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


# ---------------------------------------------------------------------------
# Scatter / Gather
# ---------------------------------------------------------------------------

class Gather(RegionCloser):
    """
    Scatter/Gather exit point.

    Stateless marker.  Pipeline detects this operator, waits for all scatter
    workers to deposit, and resumes with ``list[U]``.
    """

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        # Gather wraps T → list[T] for the operators that follow.
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


class Scatter(RegionOpener):
    """
    Scatter/Gather entry point.

    One thread passes a ``list[T]`` here; each item is dispatched to a worker
    thread that runs the scatter region independently with a fresh Context.
    The original thread blocks at the matching ``Gather`` until all workers
    have deposited their results, then resumes with ``list[U]``.

    Example::

        pipeline = Pipeline([
            ...,
            Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
            Store("tile_rects", index=1),
            Pick(0),
            Scatter(max_concurrency=4),
            Resize((640, 640)), Normalize(), Infer("model.onnx"), ..., ToDetections(),
            Gather(),
            Recall("tile_rects"),
            Stitch(iou_threshold=0.5),
        ])
    """

    closing_type = Gather

    def __init__(self, max_concurrency: int = 1) -> None:
        from .scatter import ScatterGate
        self.gate = ScatterGate(max_concurrency)

    def run_region(
        self,
        current: Any,
        label: str,
        execute_region: Callable,
        trace: Any,
        cfg: Any,
    ) -> Any:
        from .tracing import InvocationTrace, StepSpan
        from .core import _NoOpTrace
        from .tracing import merge_traces

        gate = self.gate
        collecting = isinstance(trace, InvocationTrace)
        items: list[Any] = current
        n_items = len(items)

        def run_region(entry: Any) -> None:
            child_trace = InvocationTrace(batch_size=n_items, workers=gate.max_concurrency) if collecting else _NoOpTrace()
            try:
                result, child_trace = execute_region(entry.value, child_trace)
                entry.deposit(result, child_trace if collecting else None)
            except BaseException as exc:
                entry.deposit_exception(exc, child_trace if collecting else None)

        import time
        gate.scatter(items, run_region)
        t_gather = time.perf_counter()
        try:
            entries = gate.gather()
        except BaseException:
            trace.spans.append(StepSpan(label, t_gather, time.perf_counter() - t_gather, error=True))
            raise

        child_traces = [e.child_trace for e in entries if e.child_trace is not None]
        child_trace = merge_traces(child_traces) if child_traces else None
        trace.spans.append(StepSpan(label, t_gather, time.perf_counter() - t_gather, child_trace=child_trace if collecting else None))
        return [e.result for e in entries]

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        # Scatter unwraps list[T] → T so the region sees individual items.
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (list[Any],), args[0] if args else Any
        return (list[Any],), Any


class Collate:
    """
    Stack a list of ``TensorPayload`` objects into a single batched tensor.

    Input:  ``list[TensorPayload]`` — each with shape ``(1, C, H, W)`` or
            ``(C, H, W)``.
    Output: ``TensorPayload`` with shape ``(N, C, H, W)``.
    """

    def __call__(self, tensors: list[TensorPayload]) -> TensorPayload:
        if not tensors:
            raise ValueError("Collate received an empty list")
        arrays = [t.array for t in tensors]
        if arrays[0].ndim == 4 and arrays[0].shape[0] == 1:
            # Each has a batch dim of 1 — concatenate along it.
            batched = np.concatenate(arrays, axis=0)
        else:
            batched = np.stack(arrays, axis=0)
        return TensorPayload(array=batched, layout=tensors[0].layout, dtype=tensors[0].dtype)


class Distribute:
    """
    Split a batched ``RuntimeOutputs`` back into a list of per-sample outputs.

    Input:  ``RuntimeOutputs`` — each tensor has shape ``(N, ...)``.
    Output: ``list[RuntimeOutputs]`` of length N, each with shape ``(1, ...)``.
    """

    def __call__(self, outputs: RuntimeOutputs) -> list[RuntimeOutputs]:
        n = outputs.tensors[0].array.shape[0]
        result = []
        for i in range(n):
            sample_tensors = tuple(
                TensorPayload(
                    array=t.array[i : i + 1].copy(),
                    layout=t.layout,
                    dtype=t.dtype,
                )
                for t in outputs.tensors
            )
            result.append(RuntimeOutputs(tensors=sample_tensors, names=outputs.names))
        return result

