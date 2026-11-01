from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from ml_pipes.control import SHORT_CIRCUIT
from ml_pipes.core import Pipeline
from ml_pipes.inspection import InspectionResult, PipelineInspector, TextBlock
from ml_pipes.standard import (
    Batch, CollectItems, Gather, LazyPerItem, PerItem, Scatter, StreamItems, UnBatch,
)


def double(value):
    return value * 2


def double_batch(values):
    return [value * 2 for value in values]


@pytest.mark.parametrize("operators, value, expected, closer", [
    ([Batch(1), double_batch, UnBatch()], 3, 6, UnBatch),
    ([Scatter(max_concurrency=2), double, Gather()], [1, 2, 3], [2, 4, 6], Gather),
    ([PerItem(), double, CollectItems()], [1, 2, 3], [2, 4, 6], CollectItems),
    ([Scatter(), double, Gather()], [], [], Gather),
    ([PerItem(), double, CollectItems()], [], [], CollectItems),
])
def test_inspection_captures_region_closer_and_displays_it_after_children(operators, value, expected, closer):
    result = Pipeline(operators).inspect(value)
    end = result.spans[-1]

    assert end.label == f"2:{closer.__name__}"
    assert end.operator_type is closer
    assert end.output_value == expected
    assert end.duration_s == 0.0
    assert not end.error
    views = PipelineInspector().build_views(result)
    assert views[-1].label == end.label
    assert views[-1].blocks
    html = PipelineInspector().render(result)
    if value:
        assert html.index(end.label) > html.index("1:double")
    else:
        assert end.label in html


def test_region_closer_snapshot_precedes_downstream_mutations_and_survives_serialization(tmp_path):
    closer = CollectItems()
    closer.description = "collected outputs"

    def mutate(values):
        values.append(99)
        return values

    result = Pipeline([PerItem(), double, closer, mutate]).inspect([1, 2])

    assert result.spans[2].output_value == [2, 4]
    assert result.spans[2].operator_config == {"description": "collected outputs"}
    assert result.spans[3].output_value == [2, 4, 99]
    loaded = InspectionResult.load(result.dump(tmp_path / "regions.pkl"))
    assert loaded.spans[2].label == "2:CollectItems"
    assert loaded.spans[2].output_value == [2, 4]
    assert "2:CollectItems" in PipelineInspector().render(loaded)


def test_nested_region_closers_are_captured_at_their_own_boundary():
    result = Pipeline([
        PerItem(), PerItem(), double, CollectItems(), CollectItems(),
    ]).inspect([[1, 2], [3]])

    assert [span.label for span in result.spans] == ["Input", "0:PerItem", "4:CollectItems"]
    assert result.spans[-1].output_value == [[2, 4], [6]]
    children = result.spans[1].child_trace.spans
    assert [span.label for span in children] == ["1:PerItem", "3:CollectItems"]
    assert children[-1].output_value == [2, 4]


@pytest.mark.parametrize("consume", [False, True])
def test_lazy_region_closer_does_not_consume_stream(consume):
    seen = []

    def record(value):
        seen.append(value)
        return value * 2

    operators = [LazyPerItem(), record, StreamItems()]
    if consume:
        def collect(values):
            return list(values)

        operators.append(collect)
    result = Pipeline(operators).inspect([1, 2, 3])

    assert seen == ([1, 2, 3] if consume else [])
    assert [span.label for span in result.spans] == (
        ["Input", "0:LazyPerItem", "2:StreamItems", "3:collect"]
        if consume else ["Input", "0:LazyPerItem", "2:StreamItems"]
    )
    if not consume:
        opener = result.spans[1]
        assert opener.duration_s == 0.0
        assert opener.attributes == {}
        assert opener.child_trace is None
        view = PipelineInspector().build_views(result)[1]
        assert view.blocks == [TextBlock("LazyPerItem", [("status", "child trace unavailable")])]
    end = result.spans[2]
    assert end.label == "2:StreamItems"
    assert end.attributes == {}
    assert isinstance(end.output_value, str)
    assert "_MeasuredIterable" in end.output_value
    assert "2:StreamItems" in PipelineInspector().render(result)


def test_unbatch_captures_each_invocations_own_output():
    pipeline = Pipeline([Batch(2, timeout=2), double_batch, UnBatch()])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(pipeline.inspect, [3, 7]))

    assert [result.spans[-1].output_value for result in results] == [6, 14]
    for result in results:
        assert result.spans[-1].label == "2:UnBatch"
        assert sorted(result.spans[2].child_trace.spans[0].output_value) == [6, 14]


@pytest.mark.parametrize("opener, closer", [(PerItem(), CollectItems()), (Scatter(), Gather())])
def test_region_closer_only_captures_emitted_items(opener, closer):
    def filter_and_double(value):
        return SHORT_CIRCUIT if value == 2 else value * 2

    result = Pipeline([opener, filter_and_double, closer]).inspect([1, 2, 3])

    assert result.spans[-1].output_value == [2, 6]


@pytest.mark.parametrize("opener, closer", [
    (PerItem(), CollectItems()),
    (Scatter(max_concurrency=2), Gather()),
])
def test_failed_region_preserves_child_error_without_successful_closer(opener, closer):
    def fail_on_two(value):
        if value == 2:
            raise ValueError("bad item")
        return value

    result = Pipeline([opener, fail_on_two, double, closer]).inspect([1, 2, 3])

    assert [span.label for span in result.spans] == ["Input", f"0:{type(opener).__name__}"]
    region = result.spans[1]
    assert region.error
    assert region.child_trace is not None
    assert any(span.label == "1:fail_on_two" and span.error for span in region.child_trace.spans)
