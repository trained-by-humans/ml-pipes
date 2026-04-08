from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ml_pipes import (
    DecodeOp,
    InferOp,
    LogDetectionsOp,
    MapToObjectsOp,
    NormalizeOp,
    Pipeline,
    ProjectToInputOp,
    Recall,
    ResizeOp,
    RuntimeOutputs,
    Select,
    Store,
)
from common import (
    COCO_IMAGE_NAME,
    COCO_IMAGE_URL,
    build_output_path,
    download_if_missing,
    render_and_save_detections,
)
from ml_pipes.types import DetectionBatch, TensorPayload


MODEL_URL = "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx"
MODEL_NAME = "rfdetr_nano.onnx"

RFDETR_BOX_NAMES = ("pred_boxes", "boxes", "dets")
RFDETR_LOGIT_NAMES = ("pred_logits", "logits", "labels")


def build_pipeline(model_path: Path) -> Pipeline:
    return Pipeline(
        [
            DecodeOp(),
            ResizeOp(
                size=(640, 640),
                mode="resize",
                interpolation="linear",
            ),
            Store("resize_transform", index=1),
            Select(0),
            NormalizeOp(
                output_dtype="float32",
                scale=1.0 / 255.0,
                output_layout="NCHW",
                output_color_space="RGB",
                add_batch_dim=True,
            ),
            InferOp(model_path, expected_input_layout="NCHW"),
            lambda runtime_outputs: decode_rfdetr_outputs(
                runtime_outputs,
                input_size=(640, 640),
                score_threshold=0.25,
                max_detections=20,
            ),
            Recall("resize_transform"),
            ProjectToInputOp(),
        ]
    )


def decode_rfdetr_outputs(
    runtime_outputs: RuntimeOutputs,
    input_size: tuple[int, int],
    score_threshold: float = 0.25,
    max_detections: int = 20,
) -> DetectionBatch:
    box_tensor = _select_rfdetr_tensor(runtime_outputs, RFDETR_BOX_NAMES, expected_last_dim=4)
    logit_tensor = _select_rfdetr_tensor(runtime_outputs, RFDETR_LOGIT_NAMES, expected_last_dim=None)

    boxes = _squeeze_batch_dim(box_tensor.array, "boxes")
    logits = _squeeze_batch_dim(logit_tensor.array, "logits")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"RF-DETR boxes must have shape (N, 4), got {boxes.shape}")
    if logits.ndim != 2:
        raise ValueError(f"RF-DETR logits must have shape (N, C), got {logits.shape}")
    if boxes.shape[0] != logits.shape[0]:
        raise ValueError(
            f"RF-DETR boxes/logits count mismatch: {boxes.shape[0]} boxes vs {logits.shape[0]} logits"
        )

    probabilities = _softmax(logits)
    classes = np.argmax(probabilities, axis=1).astype(np.int32)
    scores = probabilities[np.arange(probabilities.shape[0]), classes].astype(np.float32)

    boxes_xyxy = _rfdetr_boxes_to_xyxy(boxes.astype(np.float32), input_size=input_size)
    keep = scores >= score_threshold
    kept_indices = np.where(keep)[0]
    if kept_indices.size > max_detections:
        ordered = kept_indices[np.argsort(scores[kept_indices])[::-1]]
        kept_indices = ordered[:max_detections]

    return DetectionBatch(
        boxes=boxes_xyxy[kept_indices],
        scores=scores[kept_indices],
        classes=classes[kept_indices],
    )


def _select_rfdetr_tensor(
    runtime_outputs: RuntimeOutputs,
    candidate_names: tuple[str, ...],
    expected_last_dim: int | None,
) -> TensorPayload:
    for name in candidate_names:
        if name in runtime_outputs.names:
            return runtime_outputs.tensors[runtime_outputs.names.index(name)]

    matching_tensors: list[TensorPayload] = []
    for tensor in runtime_outputs.tensors:
        if tensor.array.ndim < 2:
            continue
        if expected_last_dim is not None and tensor.array.shape[-1] != expected_last_dim:
            continue
        if expected_last_dim is None and tensor.array.shape[-1] == 4:
            continue
        matching_tensors.append(tensor)

    if len(matching_tensors) == 1:
        return matching_tensors[0]

    raise ValueError(
        f"Could not resolve RF-DETR output from names {runtime_outputs.names}. "
        f"Expected one of {candidate_names}."
    )


def _squeeze_batch_dim(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported RF-DETR {label} shape: {array.shape}")


def _rfdetr_boxes_to_xyxy(boxes: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    input_h, input_w = input_size
    absolute_boxes = boxes.copy()

    # RF-DETR exports typically use normalized cx, cy, w, h boxes.
    if np.max(np.abs(absolute_boxes)) <= 2.0:
        absolute_boxes[:, [0, 2]] *= float(input_w)
        absolute_boxes[:, [1, 3]] *= float(input_h)

    centers = absolute_boxes[:, :2]
    sizes = absolute_boxes[:, 2:4]
    half_sizes = sizes / 2.0
    top_left = centers - half_sizes
    bottom_right = centers + half_sizes
    return np.concatenate((top_left, bottom_right), axis=1)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RF-DETR ONNX demo on a public COCO image.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(".example_assets"),
        help="Directory used to cache the downloaded public model and image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the annotated image. Defaults to the input image name with the model name as suffix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets_dir = args.assets_dir
    model_path = assets_dir / MODEL_NAME
    image_path = assets_dir / COCO_IMAGE_NAME
    output_path = args.output or build_output_path(assets_dir, COCO_IMAGE_NAME, MODEL_NAME)

    print(f"Downloading model to {model_path} if needed...", file=sys.stderr)
    download_if_missing(MODEL_URL, model_path)

    print(f"Downloading image to {image_path} if needed...", file=sys.stderr)
    download_if_missing(COCO_IMAGE_URL, image_path)

    pipeline = build_pipeline(model_path)
    result = pipeline(image_path)
    render_and_save_detections(
        image_path=image_path,
        detections=result,
        output_path=output_path,
    )
    Pipeline(
        [
            MapToObjectsOp(
                field_sources={
                    "box": "boxes",
                    "score": "scores",
                    "class_id": "classes",
                }
            ),
            LogDetectionsOp(
                model_path=model_path,
                image_path=image_path,
                annotated_image_path=output_path,
            ),
        ]
    )(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
