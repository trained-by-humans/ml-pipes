from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, cast

import numpy as np

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from common import decode, visualize_detections_and_store
from run_yolo8_batch import build_pipeline
from run_yolo8_onnx import yolo8_inference_pipeline

from ml_pipes import (
    AsType,
    CreateTensorMask,
    Detections,
    FilterPredictions,
    FilterPredictionsByClass,
    FilterTensors,
    ImagePayload,
    LogDetections,
    MapPredictionsToObjects,
    MapTensor,
    Pipeline,
    Pick,
    SaveImage,
    Segmentations,
    SideEffectOp,
    TensorPayload,
    TensorRegistry,
    pipeline_factory,
)


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class StringEffect(SideEffectOp[str]):
    def effect(self, payload: str) -> None:
        del payload


def build_linear() -> Pipeline[int, float]:
    return Pipeline([IntToString(), StringToFloat()])


lhs: Pipeline[int, str] = Pipeline([IntToString()])
rhs: Pipeline[str, float] = Pipeline([StringToFloat()])
assert_type(lhs + rhs, Pipeline[int, float])
assert_type(lhs >> rhs, Pipeline[int, float])
assert_type((lhs + rhs)(1), float)

linear = build_linear()
assert_type(linear, Pipeline[int, float])
assert_type(linear(1), float)

sample_image = cast(ImagePayload, None)
sample_detections = cast(Detections, None)
sample_segmentations = cast(Segmentations, None)
sample_tensor = cast(TensorPayload, None)
sample_tensor_list = cast(list[TensorPayload], None)
sample_registry = cast(TensorRegistry, None)

string_effect = StringEffect()
assert_type(string_effect("x"), str)

pick_first: Pick[Literal[0]] = Pick(0)
assert_type(pick_first((1, "value")), int)

pick_second: Pick[Literal[1]] = Pick(1)
assert_type(pick_second((1, "value")), str)

detection_filter: FilterPredictions[Detections] = FilterPredictions(
    lambda prediction: [score > 0.5 for score in prediction.scores]
)
assert_type(detection_filter(sample_detections), Detections)
assert_type(FilterPredictionsByClass({0})(sample_segmentations), Segmentations)

assert_type(AsType("float16")(sample_tensor), TensorPayload)
assert_type(AsType("float16")(sample_tensor_list), list[TensorPayload])
assert_type(AsType("float32", src="scores")(sample_registry), TensorRegistry)
assert_type(
    CreateTensorMask(
        "scores",
        predicate=lambda tensor: assert_type(tensor, np.ndarray) >= 0.5,
        as_="keep",
    ),
    CreateTensorMask,
)
assert_type(
    FilterTensors(
        "scores",
        by="classes",
        predicate=lambda classes: assert_type(classes, np.ndarray) == 0,
    ),
    FilterTensors,
)
assert_type(
    MapTensor(
        "labels",
        fn=lambda tensor: assert_type(tensor, np.ndarray).astype(np.int32),
        as_="classes",
    ),
    MapTensor,
)
assert_type(
    MapPredictionsToObjects(
        fields={
            "box": "boxes",
            "score": "scores",
            "class_id": "classes",
        }
    )(sample_detections),
    list[dict[str, object]],
)
map_to_objects_at_one: MapPredictionsToObjects[Literal[1]] = MapPredictionsToObjects(
    fields={
        "box": "boxes",
        "score": "scores",
        "class_id": "classes",
    },
    at=1,
)
assert_type(
    map_to_objects_at_one((sample_image, sample_detections)),
    tuple[ImagePayload, list[dict[str, object]]],
)

save_image: SaveImage[ImagePayload] = SaveImage(Path("result.png"))
assert_type(save_image(sample_image), ImagePayload)

save_image_at_zero: SaveImage[tuple[ImagePayload, Detections]] = SaveImage(Path("result.png"), at=0)
assert_type(save_image_at_zero((sample_image, sample_detections)), tuple[ImagePayload, Detections])

objects = cast(list[dict[str, object]], None)
log_detections: LogDetections[list[dict[str, object]]] = LogDetections("model.onnx", "image.jpg", "result.png")
assert_type(log_detections(objects), list[dict[str, object]])

log_detections_at_one: LogDetections[tuple[str, list[dict[str, object]]]] = LogDetections(
    "model.onnx",
    "image.jpg",
    "result.png",
    at=1,
)
assert_type(
    log_detections_at_one(("prefix", objects)),
    tuple[str, list[dict[str, object]]],
)

decoded = decode()
assert_type(decoded, Pipeline[str | Path, ImagePayload])

infer_pipe = yolo8_inference_pipeline(Path("model.onnx"))
assert_type(infer_pipe, Pipeline[ImagePayload, Detections])

full = decode() + infer_pipe + visualize_detections_and_store(Path("result.png"))
assert_type(full, Pipeline[str | Path, tuple[ImagePayload, Detections]])


@pipeline_factory
def make_pipeline() -> Pipeline[int, float]:
    return build_linear()


factory = make_pipeline
assert_type(factory(), Pipeline[int, float])
assert_type(factory.build({}), Pipeline[int, float])

batch_pipeline = build_pipeline(Path("model.onnx"), batch_size=4, timeout=0.05)
assert_type(batch_pipeline, Pipeline[str | Path, Detections])

with ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(batch_pipeline, Path("image.jpg"))
    assert_type(future, Future[Detections])
    detections = future.result()
    assert_type(detections, Detections)
    _ = detections.boxes
