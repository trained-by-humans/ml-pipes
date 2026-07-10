"""Tests for ScatterGate, Scatter/Gather operators, and engine integration."""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from ml_pipes.core import Pipeline
from ml_pipes.standard import (
    DropNull,
    Gather,
    Recall,
    Scatter,
    Store,
)
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vision import (
    ImagePayload,
    Resize,
)
from ml_pipes.standard.scatter import ScatterGate, _ScatterEntry


# ---------------------------------------------------------------------------
# ScatterGate unit tests
# ---------------------------------------------------------------------------

def test_scatter_gate_single_item():
    gate = ScatterGate(max_concurrency=1)

    def run(entry: _ScatterEntry) -> None:
        entry.deposit(entry.value * 2)

    gate.scatter([7], run)
    entries, first_exc = gate.gather()

    assert first_exc is None
    assert [e.result for e in entries] == [14]


def test_scatter_gate_multiple_items_in_order():
    gate = ScatterGate(max_concurrency=4)

    def run(entry: _ScatterEntry) -> None:
        # Introduce artificial jitter so threads finish out of submission order.
        time.sleep(0.01 * (4 - entry.index))
        entry.deposit(entry.value * 10)

    gate.scatter([1, 2, 3, 4], run)
    entries, first_exc = gate.gather()

    assert first_exc is None
    assert [e.result for e in entries] == [10, 20, 30, 40]


def test_scatter_gate_returns_first_exception():
    gate = ScatterGate(max_concurrency=2)
    sentinel = ValueError("worker boom")

    def run(entry: _ScatterEntry) -> None:
        if entry.index == 1:
            entry.deposit_exception(sentinel)
        else:
            time.sleep(0.02)
            entry.deposit(entry.value)

    gate.scatter([1, 2], run)
    _, first_exc = gate.gather()

    assert first_exc is sentinel


def test_scatter_gate_waits_for_all_workers_before_returning_exception():
    """gather() should wait for ALL workers before returning the first exception."""
    gate = ScatterGate(max_concurrency=3)
    finished = []
    lock = threading.Lock()

    def run(entry: _ScatterEntry) -> None:
        time.sleep(0.01 * entry.index)
        with lock:
            finished.append(entry.index)
        if entry.index == 0:
            entry.deposit_exception(RuntimeError("first fails"))
        else:
            entry.deposit(entry.value)

    gate.scatter([0, 1, 2], run)
    _, first_exc = gate.gather()

    assert isinstance(first_exc, RuntimeError)
    assert sorted(finished) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Pipeline Scatter/Gather integration
# ---------------------------------------------------------------------------

class _Double:
    def __call__(self, value: int) -> int:
        return value * 2


class _StringLen:
    def __call__(self, value: str) -> int:
        return len(value)


def test_pipeline_scatter_gather_basic():
    pipeline = Pipeline([Scatter(max_concurrency=2), _Double(), Gather()])
    result = pipeline([1, 2, 3, 4])
    assert result == [2, 4, 6, 8]


def test_pipeline_scatter_gather_preserves_order():
    pipeline = Pipeline([Scatter(max_concurrency=4), _Double(), Gather()])
    items = list(range(20))
    result = pipeline(items)
    assert result == [x * 2 for x in items]


def test_pipeline_scatter_gather_single_worker():
    pipeline = Pipeline([Scatter(max_concurrency=1), _Double(), Gather()])
    assert pipeline([5]) == [10]


def test_pipeline_scatter_gather_empty_list():
    pipeline = Pipeline([Scatter(max_concurrency=2), _Double(), Gather()])
    assert pipeline([]) == []


def test_pipeline_scatter_gather_omits_short_circuited_items():
    pipeline = Pipeline([Scatter(max_concurrency=1), DropNull(), Gather()])

    assert pipeline([1, None, 2]) == [1, 2]


def test_pipeline_scatter_gather_worker_exception_propagates():
    class _Boom:
        def __call__(self, value: int) -> int:
            if value == 3:
                raise ValueError("bad item")
            return value

    pipeline = Pipeline([Scatter(max_concurrency=4), _Boom(), Gather()])
    with pytest.raises(ValueError, match="bad item"):
        pipeline([1, 2, 3, 4])


def test_scatter_inspect_preserves_failing_child_trace():
    class _Boom:
        def __call__(self, value: int) -> int:
            if value == 3:
                raise ValueError("bad item")
            return value

    result = Pipeline([Scatter(max_concurrency=2), _Boom(), Gather()]).inspect([1, 2, 3, 4])

    assert [span.label for span in result.spans] == ["0:Scatter"]
    assert result.spans[0].error
    assert result.spans[0].child_trace is not None
    assert any(span.label == "1:_Boom" and span.error for span in result.spans[0].child_trace.spans)


def test_pipeline_scatter_gather_type_contract():
    contract = Pipeline([Scatter(max_concurrency=1), _Double(), Gather()]).validate()
    assert contract is not None


def test_scatter_type_contract_uses_outer_list_input_not_inner_region_input():
    contract = Pipeline([
        Scatter(max_concurrency=1),
        Store("snapshot"),
        Resize((32, 32)),
        Gather(),
    ]).validate(inference=True)

    assert contract is not None
    assert contract.input_type == list[ImagePayload]


def test_scatter_type_contract_returns_generic_list_if_no_type_constraints():
    contract = Pipeline([
        Scatter(max_concurrency=1),
        Store("snapshot"),
        Gather(),
    ]).validate()

    assert contract is not None
    assert contract.input_type == list[Any]


# ---------------------------------------------------------------------------
# Validation: structure
# ---------------------------------------------------------------------------

def test_validate_unmatched_scatter():
    with pytest.raises(PipelineValidationError, match=r"Pipeline step 0:Scatter has no matching Gather"):
        Pipeline([Scatter(max_concurrency=1), _Double()]).validate()


def test_validate_unmatched_gather():
    with pytest.raises(PipelineValidationError, match=r"Pipeline step 1:Gather has no matching opener"):
        Pipeline([_Double(), Gather()]).validate()


def test_validate_nested_scatter_forbidden():
    with pytest.raises(
        PipelineValidationError,
        match=r"Pipeline step 1:Scatter opens a Scatter region inside 0:Scatter",
    ):
        Pipeline([
            Scatter(max_concurrency=1),
            Scatter(max_concurrency=1),
            _Double(),
            Gather(),
            Gather(),
        ]).validate()


def test_validate_batch_inside_scatter_allowed():
    from ml_pipes.standard import (
        Batch,
        UnBatch,
    )

    class _ListToList:
        def __call__(self, values: list) -> list:
            return values

    Pipeline([
        Scatter(max_concurrency=2),
        Batch(size=1),
        _ListToList(),
        UnBatch(),
        Gather(),
    ]).validate()


def test_validate_interleaved_scatter_batch_forbidden():
    """Scatter → Batch → Gather (no UnBatch before Gather) must be rejected."""
    from ml_pipes.standard import Batch

    with pytest.raises(
        PipelineValidationError,
        match=r"Pipeline step 3:Gather.*1:Batch.*must be closed with UnBatch, not Gather",
    ):
        Pipeline([
            Scatter(max_concurrency=1),
            Batch(size=1),
            _Double(),
            Gather(),   # closes Scatter but Batch is still open
        ]).validate()


def test_validate_interleaved_batch_scatter_forbidden():
    """Batch → Scatter → UnBatch (no Gather before UnBatch) must be rejected."""
    from ml_pipes.standard import (
        Batch,
        UnBatch,
    )

    with pytest.raises(
        PipelineValidationError,
        match=r"Pipeline step 3:UnBatch.*1:Scatter.*must be closed with Gather, not UnBatch",
    ):
        Pipeline([
            Batch(size=1),
            Scatter(max_concurrency=1),
            _Double(),
            UnBatch(),  # closes Batch but Scatter is still open
        ]).validate()


# ---------------------------------------------------------------------------
# Validation: context scoping
# ---------------------------------------------------------------------------

def test_scatter_context_isolates_store_from_outer():
    """Store inside scatter region is invisible after Gather."""
    with pytest.raises(PipelineValidationError, match="Recall"):
        Pipeline([
            Scatter(max_concurrency=1),
            Store("x"),
            _Double(),
            Gather(),
            Recall("x"),
        ]).validate()


def test_scatter_context_recall_inside_region_forbidden():
    """Recall inside scatter region for a key stored outside is forbidden."""
    with pytest.raises(PipelineValidationError, match="Recall"):
        Pipeline([
            Store("x"),
            Scatter(max_concurrency=1),
            Recall("x"),
            Gather(),
        ]).validate()
