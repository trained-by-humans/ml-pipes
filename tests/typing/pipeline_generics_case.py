from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from ml_pipes.core import Pipeline
from ml_pipes.factory import pipeline_factory
from ml_pipes.standard import Pick, SideEffectOp
from ml_pipes.tensor import AsType, CreateTensorMask, FilterTensors, MapTensor, TensorPayload, TensorRegistry
from ml_pipes.vision import FilterTensorsByBoxArea, ImagePayload, LogDetections, SaveImage


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class StringEffect(SideEffectOp[str]):
    def effect(self, payload: str) -> None:
        del payload


lhs: Pipeline[int, str] = Pipeline([IntToString()])
rhs: Pipeline[str, float] = Pipeline([StringToFloat()])
assert_type(lhs + rhs, Pipeline[int, float])
assert_type((lhs + rhs)(1), float)

sample_image = cast(ImagePayload, None)
sample_tensor = cast(TensorPayload, None)
sample_tensor_list = cast(list[TensorPayload], None)
sample_registry = cast(TensorRegistry, None)

assert_type(StringEffect()("x"), str)
pick_second: Pick[Literal[1]] = Pick(1)
assert_type(pick_second((1, "value")), str)
assert_type(AsType("float16")(sample_tensor), TensorPayload)
assert_type(AsType("float16")(sample_tensor_list), list[TensorPayload])
assert_type(AsType("float32", src="scores")(sample_registry), TensorRegistry)
assert_type(CreateTensorMask("scores", predicate=lambda tensor: assert_type(tensor, np.ndarray) >= 0.5, as_="keep"), CreateTensorMask)
assert_type(FilterTensors("scores", by="classes", predicate=lambda classes: assert_type(classes, np.ndarray) == 0), FilterTensors)
assert_type(MapTensor("labels", fn=lambda tensor: assert_type(tensor, np.ndarray).astype(np.int32), as_="classes"), MapTensor)
assert_type(FilterTensorsByBoxArea("scores", min_area=1.0)(sample_registry), TensorRegistry)

save_image: SaveImage[ImagePayload] = SaveImage(Path("result.png"))
assert_type(save_image(sample_image), ImagePayload)
save_image_at_zero: SaveImage[tuple[ImagePayload, TensorRegistry]] = SaveImage(Path("result.png"), at=0)
assert_type(save_image_at_zero((sample_image, sample_registry)), tuple[ImagePayload, TensorRegistry])

log_detections: LogDetections[TensorRegistry] = LogDetections("model.onnx", "image.jpg", "result.png")
assert_type(log_detections(sample_registry), TensorRegistry)
log_detections_at_one: LogDetections[tuple[str, TensorRegistry]] = LogDetections("model.onnx", "image.jpg", "result.png", at=1)
assert_type(log_detections_at_one(("prefix", sample_registry)), tuple[str, TensorRegistry])

@pipeline_factory
def make_pipeline() -> Pipeline[int, float]:
    return lhs + rhs


assert_type(make_pipeline(), Pipeline[int, float])
