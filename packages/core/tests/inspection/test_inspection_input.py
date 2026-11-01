from __future__ import annotations

import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.inspection import ImageBlock, InspectionResult, PipelineInspector


@pytest.mark.parametrize("value", [None, 42, [], {"name": "input"}])
def test_empty_pipeline_still_captures_its_input(value):
    result = Pipeline([]).inspect(value)

    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.label == "Input"
    assert span.output_value == value
    assert span.duration_s == 0.0
    assert span.operator_type is None
    assert not span.error
    assert PipelineInspector().build_views(result)[0].label == "Input"


def test_input_image_is_snapshotted_before_first_operator_and_round_trips(tmp_path):
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    def paint(value):
        value[:] = 255
        return value

    result = Pipeline([paint]).inspect(image)

    assert [span.label for span in result.spans] == ["Input", "0:paint"]
    assert result.spans[0].output_shape == "ndarray (4, 6, 3)"
    assert np.all(result.spans[0].output_value == 0)
    assert np.all(result.spans[1].output_value == 255)
    loaded = InspectionResult.load(result.dump(tmp_path / "input.pkl"))
    assert np.array_equal(loaded.spans[0].output_value, result.spans[0].output_value)
    views = PipelineInspector().build_views(loaded)
    assert views[0].label == "Input"
    assert isinstance(views[0].blocks[0], ImageBlock)
    assert np.all(views[0].blocks[0].array == 0)
    html = PipelineInspector().render(loaded)
    assert html.index('>Input<') < html.index('>0:paint<')


def test_input_is_available_when_first_operator_fails():
    def fail(value):
        value.append("mutated")
        raise ValueError("failed")

    result = Pipeline([fail]).inspect(["original"])

    assert result.spans[0].label == "Input"
    assert result.spans[0].output_value == ["original"]
    assert result.spans[1].error


def test_iterator_input_is_described_without_consuming_or_copying_it():
    class Items:
        def __init__(self):
            self.current = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.current == 3:
                raise StopIteration
            self.current += 1
            return self.current

        def __deepcopy__(self, memo):
            raise AssertionError("Iterator input should not be copied")

    def collect(values):
        return list(values)

    source = Items()
    untouched = Pipeline([]).inspect(source)
    assert source.current == 0
    assert untouched.spans[0].attributes == {}
    assert isinstance(untouched.spans[0].output_value, str)
    result = Pipeline([collect]).inspect(source)
    assert result.spans[1].output_value == [1, 2, 3]


def test_skipping_input_avoids_snapshot_work_for_empty_pipeline():
    copy_attempts = []

    class Input:
        def __deepcopy__(self, memo):
            copy_attempts.append(True)
            return Input()

    result = Pipeline([]).inspect(Input(), capture_input=False)

    assert result.spans == []
    assert copy_attempts == []
    assert PipelineInspector().build_views(result) == []


def test_skipping_input_preserves_step_snapshots_and_serialization(tmp_path):
    def append(value):
        value.append(1)
        return value

    value = []
    result = Pipeline([append, append]).inspect(value, capture_input=False)

    assert value == [1, 1]
    assert [span.label for span in result.spans] == ["0:append", "1:append"]
    assert [span.output_value for span in result.spans] == [[1], [1, 1]]
    loaded = InspectionResult.load(result.dump(tmp_path / "outputs.pkl"))
    assert [span.label for span in loaded.spans] == ["0:append", "1:append"]
    assert [span.output_value for span in loaded.spans] == [[1], [1, 1]]


def test_skipping_input_preserves_first_operator_failure():
    def fail(value):
        raise ValueError("failed")

    result = Pipeline([fail]).inspect("input", capture_input=False)

    assert len(result.spans) == 1
    assert result.spans[0].label == "0:fail"
    assert result.spans[0].error
