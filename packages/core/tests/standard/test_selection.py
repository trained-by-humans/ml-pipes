from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick, Select
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vision import ImagePayload


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class IntToPair:
    def __call__(self, value: int) -> tuple[int, str]:
        return value, str(value)


class MakeImage:
    def __call__(self, value: int) -> ImagePayload:
        return ImagePayload(array=np.zeros((10, 20, 3), dtype=np.uint8), color_space="BGR", layout="HWC")


class AcceptArray:
    def __call__(self, value: np.ndarray) -> int:
        return int(value.shape[0])


def test_pick_validation_propagates_element_type():
    pipeline = Pipeline([IntToPair(), Pick(0), IntToString()])

    pipeline.validate()


def test_pick_validation_rejects_wrong_downstream_type():
    pipeline = Pipeline([IntToPair(), Pick(0), StringToFloat()])

    with pytest.raises(PipelineValidationError, match="contract mismatch"):
        pipeline.validate()


def test_pick_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Pick(5)])

    with pytest.raises(PipelineValidationError, match="Pick\\(5\\) is out of bounds"):
        pipeline.validate()


def test_pick_negative_out_of_bounds_raises_on_concrete_input():
    pipeline = Pipeline([IntToPair(), Pick(-3)])

    with pytest.raises(PipelineValidationError, match="Pick\\(-3\\) is out of bounds"):
        pipeline.validate()


def test_pick_out_of_bounds_silent_on_vague_input():
    pipeline = Pipeline([Pick(5)])

    pipeline.validate()


def test_pick_validation_rejects_known_non_tuple_input():
    pipeline = Pipeline([IntToString(), Pick(0)])

    with pytest.raises(PipelineValidationError, match="Pick requires a tuple boundary"):
        pipeline.validate()


def test_pick_establishes_tuple_input_boundary_from_downstream_type():
    contract = Pipeline([Pick(0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type == tuple[Any, ...]


def test_select_validation_propagates_array_attribute_type():
    contract = Pipeline([MakeImage(), Select("array"), AcceptArray()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_propagates_nested_attribute_type():
    contract = Pipeline([MakeImage(), Select(("spatial_shape", 0)), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_accepts_dotted_string_selector():
    contract = Pipeline([MakeImage(), Select("spatial_shape.0"), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_validation_accepts_variadic_selector_parts():
    contract = Pipeline([MakeImage(), Select("spatial_shape", 0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is int


def test_select_tuple_index_rejects_known_non_tuple_input():
    pipeline = Pipeline([IntToString(), Select(0)])

    with pytest.raises(PipelineValidationError, match="indexable"):
        pipeline.validate()


def test_select_tuple_index_establishes_tuple_input_boundary():
    contract = Pipeline([Select(0), IntToString()]).validate()

    assert contract is not None
    assert contract.input_type is Any
