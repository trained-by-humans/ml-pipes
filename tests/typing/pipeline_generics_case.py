from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from common import decode, visualize_detections_and_store
from run_yolo8_batch import build_pipeline
from run_yolo8_onnx import yolo8_inference_pipeline

from ml_pipes import Detections, ImagePayload, Pipeline, pipeline_factory


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


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
