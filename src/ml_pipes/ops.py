from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Literal
from typing import TextIO

import numpy as np

from .transforms import ResizeTransform
from .types import (
    DetectionArrays,
    Detections,
    ImagePayload,
    RuntimeOutputs,
    SegmentationCandidates,
    Segmentations,
    TensorPayload,
)


class DecodeOp:
    def __call__(self, image_path: str | Path) -> ImagePayload:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
        return payload


class ResizeOp:
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
            raise ValueError(f"ResizeOp expects HWC image layout, got {payload.layout}")

    def _resolve_interpolation(self, cv2: object) -> int:
        mapping = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        return mapping[self.interpolation]


class NormalizeOp:
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
            raise ValueError(f"NormalizeOp expects HWC image layout, got {image_payload.layout}")

        image = image_payload.array
        if image_payload.color_space != self.output_color_space and {
            image_payload.color_space,
            self.output_color_space,
        } == {"BGR", "RGB"}:
            image = image[..., ::-1]
        elif image_payload.color_space != self.output_color_space:
            raise ValueError(
                f"NormalizeOp cannot convert {image_payload.color_space} to {self.output_color_space}"
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


class CastTensorOp:
    def __init__(self, dtype: str, selector: str | None = None):
        self.dtype = np.dtype(dtype)
        self.selector = selector

    def __call__(self, value: object) -> object:
        if self.selector is not None:
            selected = getattr(value, self.selector)
            casted = self._cast_tensor_value(selected)
            if is_dataclass(value):
                return replace(value, **{self.selector: casted})
            raise TypeError(
                f"CastTensorOp selector={self.selector!r} requires a dataclass value, got {type(value)!r}"
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
        raise TypeError(f"CastTensorOp does not support value type {type(value)!r}")


class InferOp:
    def __init__(
        self,
        model_path: str | Path,
        # Runtime-facing execution provider config.
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        # Runtime-facing input binding. Input names come from the exported graph.
        input_name: str | None = None,
        # Runtime-facing input tensor contract.
        expected_input_layout: str = "NCHW",
        # Model-facing input dtype contract.
        expected_model_dtype: str | None = None,
        # Runtime-facing output tensor metadata aligned with exported graph output order.
        output_layouts: tuple[str, ...] | None = None,
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
        self.input_name = input_name or self.session.get_inputs()[0].name
        self.expected_input_layout = expected_input_layout
        self.model_dtype = np.dtype(expected_model_dtype) if expected_model_dtype is not None else None
        self.output_layouts = output_layouts
        self.output_names = tuple(output.name for output in self.session.get_outputs())

    def __call__(self, tensor_payload: TensorPayload) -> RuntimeOutputs:
        if tensor_payload.layout != self.expected_input_layout:
            raise ValueError(
                f"InferOp expects {self.expected_input_layout} tensor layout, got {tensor_payload.layout}"
            )

        actual_dtype = np.dtype(tensor_payload.dtype)
        if self.model_dtype is not None and actual_dtype != self.model_dtype:
            raise ValueError(f"InferOp expects model dtype {self.model_dtype}, got {actual_dtype}")

        outputs = self.session.run(None, {self.input_name: tensor_payload.array})

        if self.output_layouts is None:
            output_layouts = tuple("UNKNOWN" for _ in outputs)
        else:
            if len(self.output_layouts) != len(outputs):
                raise ValueError(
                    f"InferOp expected {len(self.output_layouts)} output layouts, got {len(outputs)} outputs"
                )
            output_layouts = self.output_layouts

        runtime_output_names = self.output_names or tuple(f"output_{index}" for index in range(len(outputs)))
        tensors = tuple(
            TensorPayload(array=np.asarray(output), layout=layout, dtype=str(np.asarray(output).dtype))
            for output, layout in zip(outputs, output_layouts, strict=True)
        )
        return RuntimeOutputs(tensors=tensors, names=runtime_output_names)


class DecodePredictionsOp:
    def __init__(
        self,
        # Export/model-facing output selection. Names and indexes come from the exported model artifact.
        export_output_index: int = 0,
        export_output_name: str | None = None,
        num_box_values: int = 4,
        class_start_index: int = 4,
        input_box_format: Literal["xywh", "xyxy"] = "xywh",
        transpose_output: Literal["auto", "never", "always"] = "auto",
        squeeze_batch_dim: bool = True,
        score_activation: Literal["none", "sigmoid", "softmax"] = "none",
    ):
        self.export_output_index = export_output_index
        self.export_output_name = export_output_name
        self.num_box_values = num_box_values
        self.class_start_index = class_start_index
        self.input_box_format = input_box_format
        self.transpose_output = transpose_output
        self.squeeze_batch_dim = squeeze_batch_dim
        self.score_activation = score_activation

    def __call__(self, runtime_outputs: RuntimeOutputs) -> DetectionArrays:
        export_output = self._select_export_output(runtime_outputs)
        predictions = self._normalize_output_shape(np.asarray(export_output.array))
        if predictions.shape[1] < self.class_start_index + 2:
            raise ValueError(
                "Unsupported YOLOv8 output: expected at least 4 box values and 2 class scores"
            )

        boxes = predictions[:, : self.num_box_values]
        class_scores = predictions[:, self.class_start_index :]
        class_scores = self._activate_scores(class_scores)
        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]
        boxes_xyxy = self._to_xyxy(boxes)
        batch = DetectionArrays(
            boxes=boxes_xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            classes=classes,
        )
        return batch

    def _select_export_output(self, value: RuntimeOutputs) -> TensorPayload:
        if self.export_output_name is not None:
            if self.export_output_name not in value.names:
                raise ValueError(
                    f"DecodePredictionsOp export output {self.export_output_name!r} not found in {value.names}"
                )
            return value.tensors[value.names.index(self.export_output_name)]

        if self.export_output_index >= len(value.tensors):
            raise ValueError(
                f"DecodePredictionsOp export output index {self.export_output_index} out of range "
                f"for {len(value.tensors)} runtime outputs"
            )
        return value.tensors[self.export_output_index]

    def _normalize_output_shape(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 3 and self.squeeze_batch_dim and output.shape[0] == 1:
            output = output[0]
        elif output.ndim != 2:
            raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

        if output.ndim != 2:
            raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

        if self.transpose_output == "never":
            return output
        if self.transpose_output == "always":
            return output.T
        if output.shape[0] > output.shape[1] and output.shape[1] >= self.class_start_index + 2:
            return output
        if output.shape[0] >= self.class_start_index + 2:
            return output.T
        raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

    def _to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        if self.input_box_format == "xyxy":
            return boxes
        if self.input_box_format != "xywh":
            raise ValueError(f"Unsupported box format: {self.input_box_format}")

        centers = boxes[:, :2]
        sizes = boxes[:, 2:4]
        half_sizes = sizes / 2.0
        top_left = centers - half_sizes
        bottom_right = centers + half_sizes
        return np.concatenate((top_left, bottom_right), axis=1)

    def _activate_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.score_activation == "none":
            return scores
        if self.score_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-scores))
        if self.score_activation == "softmax":
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / np.sum(exp, axis=1, keepdims=True)
        raise ValueError(f"Unsupported score activation: {self.score_activation}")


class NMSOp:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def __call__(self, detection_arrays: DetectionArrays) -> DetectionArrays:
        detections = detection_arrays
        boxes, scores, classes, kept = self._filter_and_keep_indices(
            detections.boxes,
            detections.scores,
            detections.classes,
        )

        if boxes.size == 0:
            empty = DetectionArrays(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                classes=np.zeros((0,), dtype=np.int32),
            )
            return empty

        filtered = DetectionArrays(
            boxes=boxes[kept],
            scores=scores[kept],
            classes=classes[kept],
        )
        return filtered

    def _filter_and_keep_indices(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        if boxes.size == 0:
            return boxes, scores, classes, np.zeros((0,), dtype=np.int32)

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

        kept = np.asarray(kept_indices, dtype=np.int32)
        final_order = np.argsort(scores[kept])[::-1]
        kept = kept[final_order][: self.max_detections]
        return boxes, scores, classes, kept

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


class ProjectToInputOp:
    def __call__(self, value: DetectionArrays, transform: ResizeTransform) -> Detections:
        boxes = value.boxes.copy()
        scores = value.scores.astype(np.float32)
        classes = value.classes.astype(np.int32)

        pad_x, pad_y = transform.pad
        scale_x, scale_y = transform.scale
        original_h, original_w = transform.original_shape

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))

        return Detections(
            boxes=boxes.tolist(),
            scores=scores.tolist(),
            classes=classes.tolist(),
        )


class DecodeSegmentationOp:
    def __init__(
        self,
        export_detection_output_index: int = 0,
        export_detection_output_name: str | None = None,
        export_prototype_output_index: int = 1,
        export_prototype_output_name: str | None = None,
        num_box_values: int = 4,
        class_start_index: int = 4,
        num_masks: int = 32,
        input_box_format: Literal["xywh", "xyxy"] = "xywh",
        transpose_output: Literal["auto", "never", "always"] = "auto",
        squeeze_batch_dim: bool = True,
        score_activation: Literal["none", "sigmoid", "softmax"] = "none",
    ):
        self.export_detection_output_index = export_detection_output_index
        self.export_detection_output_name = export_detection_output_name
        self.export_prototype_output_index = export_prototype_output_index
        self.export_prototype_output_name = export_prototype_output_name
        self.num_box_values = num_box_values
        self.class_start_index = class_start_index
        self.num_masks = num_masks
        self.input_box_format = input_box_format
        self.transpose_output = transpose_output
        self.squeeze_batch_dim = squeeze_batch_dim
        self.score_activation = score_activation

    def __call__(self, runtime_outputs: RuntimeOutputs) -> SegmentationCandidates:
        detection_output = self._select_export_output(
            runtime_outputs,
            output_index=self.export_detection_output_index,
            output_name=self.export_detection_output_name,
        )
        prototype_output = self._select_export_output(
            runtime_outputs,
            output_index=self.export_prototype_output_index,
            output_name=self.export_prototype_output_name,
        )

        predictions = self._normalize_prediction_shape(np.asarray(detection_output.array))
        prototypes = self._normalize_prototype_shape(np.asarray(prototype_output.array))
        expected_columns = self.class_start_index + 1 + self.num_masks
        if predictions.shape[1] < expected_columns:
            raise ValueError(
                "Unsupported YOLO-seg output: expected box, class, and mask coefficient columns"
            )

        boxes = predictions[:, : self.num_box_values]
        mask_coefficients = predictions[:, -self.num_masks :]
        class_scores = predictions[:, self.class_start_index : -self.num_masks]
        class_scores = self._activate_scores(class_scores)
        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]
        return SegmentationCandidates(
            boxes=self._to_xyxy(boxes).astype(np.float32),
            scores=scores.astype(np.float32),
            classes=classes,
            mask_coefficients=mask_coefficients.astype(np.float32),
            prototypes=prototypes.astype(np.float32),
        )

    def _select_export_output(
        self,
        value: RuntimeOutputs,
        *,
        output_index: int,
        output_name: str | None,
    ) -> TensorPayload:
        if output_name is not None:
            if output_name not in value.names:
                raise ValueError(f"Segmentation export output {output_name!r} not found in {value.names}")
            return value.tensors[value.names.index(output_name)]
        if output_index >= len(value.tensors):
            raise ValueError(
                f"Segmentation export output index {output_index} out of range for {len(value.tensors)} runtime outputs"
            )
        return value.tensors[output_index]

    def _normalize_prediction_shape(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 3 and self.squeeze_batch_dim and output.shape[0] == 1:
            output = output[0]
        elif output.ndim != 2:
            raise ValueError(f"Unsupported YOLO-seg detection output shape: {output.shape}")

        if self.transpose_output == "never":
            return output
        if self.transpose_output == "always":
            return output.T
        if output.shape[0] > output.shape[1] and output.shape[1] >= self.class_start_index + 1 + self.num_masks:
            return output
        if output.shape[0] >= self.class_start_index + 1 + self.num_masks:
            return output.T
        raise ValueError(f"Unsupported YOLO-seg detection output shape: {output.shape}")

    def _normalize_prototype_shape(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 4 and self.squeeze_batch_dim and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 3:
            raise ValueError(f"Unsupported YOLO-seg prototype output shape: {output.shape}")
        if output.shape[0] != self.num_masks:
            raise ValueError(
                f"YOLO-seg prototype output expected {self.num_masks} mask channels, got {output.shape[0]}"
            )
        return output

    def _to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        if self.input_box_format == "xyxy":
            return boxes
        if self.input_box_format != "xywh":
            raise ValueError(f"Unsupported box format: {self.input_box_format}")
        centers = boxes[:, :2]
        sizes = boxes[:, 2:4]
        half_sizes = sizes / 2.0
        top_left = centers - half_sizes
        bottom_right = centers + half_sizes
        return np.concatenate((top_left, bottom_right), axis=1)

    def _activate_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.score_activation == "none":
            return scores
        if self.score_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-scores))
        if self.score_activation == "softmax":
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp = np.exp(shifted)
            return exp / np.sum(exp, axis=1, keepdims=True)
        raise ValueError(f"Unsupported score activation: {self.score_activation}")


class SegmentationNMSOp(NMSOp):
    def __call__(self, value: SegmentationCandidates) -> SegmentationCandidates:
        boxes, scores, classes, kept = self._filter_and_keep_indices(
            value.boxes,
            value.scores,
            value.classes,
        )
        if boxes.size == 0:
            return SegmentationCandidates(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                classes=np.zeros((0,), dtype=np.int32),
                mask_coefficients=np.zeros((0, value.mask_coefficients.shape[1]), dtype=np.float32),
                prototypes=value.prototypes,
            )
        return SegmentationCandidates(
            boxes=boxes[kept],
            scores=scores[kept],
            classes=classes[kept],
            mask_coefficients=value.mask_coefficients[value.scores >= self.conf_threshold][kept],
            prototypes=value.prototypes,
        )


class ProjectSegmentationsOp:
    def __init__(self, mask_threshold: float = 0.0):
        self.mask_threshold = mask_threshold

    def __call__(self, value: SegmentationCandidates, transform: ResizeTransform) -> Segmentations:
        boxes = value.boxes.copy()
        scores = value.scores.astype(np.float32)
        classes = value.classes.astype(np.int32)

        masks = self._process_masks(
            value.prototypes,
            value.mask_coefficients,
            boxes,
            transform.resized_shape,
            transform,
        )
        valid = np.asarray([mask.any() for mask in masks], dtype=bool)
        boxes = boxes[valid]
        scores = scores[valid]
        classes = classes[valid]
        masks = [mask for mask, keep in zip(masks, valid, strict=True) if keep]

        pad_x, pad_y = transform.pad
        scale_x, scale_y = transform.scale
        original_h, original_w = transform.original_shape

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))

        return Segmentations(
            boxes=boxes.tolist(),
            scores=scores.tolist(),
            classes=classes.tolist(),
            masks=masks,
        )

    def _process_masks(
        self,
        prototypes: np.ndarray,
        mask_coefficients: np.ndarray,
        boxes: np.ndarray,
        resized_shape: tuple[int, int],
        transform: ResizeTransform,
    ) -> list[np.ndarray]:
        import cv2

        channels, mask_h, mask_w = prototypes.shape
        masks = mask_coefficients @ prototypes.reshape(channels, -1)
        masks = masks.reshape(-1, mask_h, mask_w)

        width_ratio = mask_w / resized_shape[1]
        height_ratio = mask_h / resized_shape[0]
        downsampled_boxes = boxes.copy()
        downsampled_boxes[:, [0, 2]] *= width_ratio
        downsampled_boxes[:, [1, 3]] *= height_ratio
        masks = self._crop_masks(masks, downsampled_boxes)

        projected: list[np.ndarray] = []
        for mask in masks:
            upsampled = cv2.resize(mask, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_LINEAR)
            scaled = self._scale_mask_to_input(upsampled, transform)
            projected.append((scaled > self.mask_threshold).astype(np.uint8))
        return projected

    def _scale_mask_to_input(self, mask: np.ndarray, transform: ResizeTransform) -> np.ndarray:
        import cv2

        resized_h, resized_w = transform.resized_shape
        pad_x, pad_y = transform.pad
        top = max(int(round(pad_y - 0.1)), 0)
        left = max(int(round(pad_x - 0.1)), 0)
        bottom = min(int(round(resized_h - pad_y + 0.1)), resized_h)
        right = min(int(round(resized_w - pad_x + 0.1)), resized_w)
        mask = mask[top:bottom, left:right]
        original_h, original_w = transform.original_shape
        return cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    def _crop_masks(self, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        num_masks, height, width = masks.shape
        x1 = boxes[:, 0].clip(0, width)
        y1 = boxes[:, 1].clip(0, height)
        x2 = boxes[:, 2].clip(0, width)
        y2 = boxes[:, 3].clip(0, height)

        rows = np.arange(width, dtype=np.float32)[None, None, :]
        cols = np.arange(height, dtype=np.float32)[None, :, None]
        return masks * ((rows >= x1[:, None, None]) * (rows < x2[:, None, None]) * (cols >= y1[:, None, None]) * (cols < y2[:, None, None]))


class DrawBoxesOp:
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

    def __call__(self, detections: Detections, source_image: ImagePayload) -> ImagePayload:
        import cv2

        if source_image is None:
            raise ValueError("source_image missing from context; cannot draw detections")

        image = np.asarray(source_image).copy()
        detections = detections
        boxes = detections.boxes
        scores = detections.scores
        classes = detections.classes

        for box, score, class_id in zip(boxes, scores, classes, strict=True):
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

        return ImagePayload(array=image, color_space="BGR", layout="HWC")

    def _format_label(self, class_id: int, score: float) -> str:
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            name = self.class_names[class_id]
        else:
            name = str(class_id)
        return f"{name} {score:.2f}"


class SaveImageOp:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def __call__(self, image_payload: ImagePayload) -> ImagePayload:
        import cv2

        if image_payload.layout != "HWC":
            raise ValueError(f"SaveImageOp expects HWC image layout, got {image_payload.layout}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(self.output_path), image_payload.array)
        if not written:
            raise ValueError(f"Failed to write image: {self.output_path}")
        return image_payload


class MapToObjectsOp:
    def __init__(
        self,
        field_sources: dict[str, str | Callable[[object], Sequence[object]]],
    ):
        self.field_sources = field_sources

    def __call__(self, prediction_arrays: object) -> list[dict[str, object]]:
        columns: dict[str, Sequence[object]] = {}
        for field_name, source in self.field_sources.items():
            if isinstance(source, str):
                column = getattr(prediction_arrays, source)
            else:
                column = source(prediction_arrays)
            columns[field_name] = column

        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"CollectionsToObjectsOp requires equal-length collections, got lengths {sorted(lengths)}")

        records: list[dict[str, object]] = []
        field_names = tuple(columns.keys())
        rows = zip(*(columns[field_name] for field_name in field_names), strict=True)
        for row in rows:
            record = dict(zip(field_names, row, strict=True))
            records.append(record)
        return records


class LogDetectionsOp:
    def __init__(
        self,
        model_path: str | Path,
        image_path: str | Path,
        annotated_image_path: str | Path,
        indent: int = 2,
        stream: TextIO | None = None,
    ):
        self.model_path = Path(model_path)
        self.image_path = Path(image_path)
        self.annotated_image_path = Path(annotated_image_path)
        self.indent = indent
        self.stream = stream or sys.stdout

    def __call__(self, prediction_objects: list[dict[str, object]]) -> list[dict[str, object]]:
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
        return prediction_objects
