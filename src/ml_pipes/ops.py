from __future__ import annotations

from pathlib import Path

import numpy as np

from .transforms import ResizeTransform
from .types import DetectionBatch, DetectionResult, ImagePayload, TensorPayload


class DecodeOp:
    def __call__(self, value: str | Path) -> ImagePayload:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {path}")
        payload = ImagePayload(array=image, color_space="BGR", layout="HWC")
        return payload


class ResizeOp:
    def __init__(self, size: tuple[int, int] = (640, 640), pad_value: int = 114):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, value: ImagePayload) -> tuple[ImagePayload, ResizeTransform]:
        import cv2

        self._validate_image_payload(value)
        image = value.array
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.size
        scale = min(target_h / original_h, target_w / original_w)

        resized_w = int(round(original_w * scale))
        resized_h = int(round(original_h * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        dw = target_w - resized_w
        dh = target_h - resized_h
        left = int(np.floor(dw / 2))
        right = int(np.ceil(dw / 2))
        top = int(np.floor(dh / 2))
        bottom = int(np.ceil(dh / 2))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )

        transform = ResizeTransform(
            scale=scale,
            pad=(float(left), float(top)),
            original_shape=(original_h, original_w),
        )
        payload = ImagePayload(
            array=padded,
            color_space=value.color_space,
            layout=value.layout,
        )
        return payload, transform

    @staticmethod
    def _validate_image_payload(payload: ImagePayload) -> None:
        if payload.layout != "HWC":
            raise ValueError(f"ResizeOp expects HWC image layout, got {payload.layout}")


class NormalizeOp:
    def __call__(self, value: ImagePayload) -> TensorPayload:
        if value.layout != "HWC":
            raise ValueError(f"NormalizeOp expects HWC image layout, got {value.layout}")

        image = value.array
        if value.color_space == "BGR":
            image = image[..., ::-1]
        elif value.color_space != "RGB":
            raise ValueError(f"NormalizeOp expects BGR or RGB image payload, got {value.color_space}")

        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        payload = TensorPayload(array=tensor, layout="NCHW", dtype="float32")
        return payload


class InferOp:
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

        import onnxruntime as ort

        self.model_path = path
        self.session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, value: TensorPayload) -> TensorPayload:
        if value.layout != "NCHW":
            raise ValueError(f"InferOp expects NCHW tensor layout, got {value.layout}")

        outputs = self.session.run(None, {self.input_name: value.array})
        payload = TensorPayload(array=np.asarray(outputs[0]), layout="UNKNOWN", dtype=str(outputs[0].dtype))
        return payload


class DecodePredictionsOp:
    def __call__(self, value: TensorPayload) -> DetectionBatch:
        predictions = self._normalize_output_shape(np.asarray(value.array))
        if predictions.shape[1] < 6:
            raise ValueError(
                "Unsupported YOLOv8 output: expected at least 4 box values and 2 class scores"
            )

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        batch = DetectionBatch(
            boxes=boxes_xyxy.astype(np.float32),
            scores=scores.astype(np.float32),
            classes=classes,
        )
        return batch

    @staticmethod
    def _normalize_output_shape(output: np.ndarray) -> np.ndarray:
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        elif output.ndim != 2:
            raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

        if output.ndim != 2:
            raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

        if output.shape[0] > output.shape[1] and output.shape[1] >= 6:
            return output
        if output.shape[0] >= 6:
            return output.T
        raise ValueError(f"Unsupported YOLOv8 output shape: {output.shape}")

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        centers = boxes[:, :2]
        sizes = boxes[:, 2:4]
        half_sizes = sizes / 2.0
        top_left = centers - half_sizes
        bottom_right = centers + half_sizes
        return np.concatenate((top_left, bottom_right), axis=1)


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

    def __call__(self, value: DetectionBatch) -> DetectionBatch:
        detections = value
        mask = detections.scores >= self.conf_threshold
        boxes = detections.boxes[mask]
        scores = detections.scores[mask]
        classes = detections.classes[mask]

        if boxes.size == 0:
            empty = DetectionBatch(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                classes=np.zeros((0,), dtype=np.int32),
            )
            return empty

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

        filtered = DetectionBatch(
            boxes=boxes[kept],
            scores=scores[kept],
            classes=classes[kept],
        )
        return filtered

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
    def __call__(self, value: DetectionBatch, transform: ResizeTransform) -> DetectionResult:
        boxes = value.boxes.copy()
        scores = value.scores.astype(np.float32)
        classes = value.classes.astype(np.int32)

        pad_x, pad_y = transform.pad
        scale = transform.scale
        original_h, original_w = transform.original_shape

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))

        return DetectionResult(
            boxes=boxes.tolist(),
            scores=scores.tolist(),
            classes=classes.tolist(),
        )


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

    def __call__(self, value: DetectionResult, source_image: ImagePayload) -> ImagePayload:
        import cv2

        if source_image is None:
            raise ValueError("source_image missing from context; cannot draw detections")

        image = np.asarray(source_image).copy()
        detections = value
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

    def __call__(self, value: ImagePayload) -> ImagePayload:
        import cv2

        if value.layout != "HWC":
            raise ValueError(f"SaveImageOp expects HWC image layout, got {value.layout}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(self.output_path), value.array)
        if not written:
            raise ValueError(f"Failed to write image: {self.output_path}")
        return value
