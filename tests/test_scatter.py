"""Tests for ScatterGate, Scatter/Gather operators, and engine integration."""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from ml_pipes import Gather, Pipeline, PipelineValidationError, Recall, Scatter, Store
from ml_pipes.scatter import ScatterGate, _ScatterEntry


# ---------------------------------------------------------------------------
# ScatterGate unit tests
# ---------------------------------------------------------------------------

def test_scatter_gate_single_item():
    gate = ScatterGate(max_concurrency=1)
    results: list[Any] = []

    def run(entry: _ScatterEntry) -> None:
        entry.deposit(entry.value * 2)

    gate.scatter([7], run)
    assert [e.result for e in gate.gather()] == [14]


def test_scatter_gate_multiple_items_in_order():
    gate = ScatterGate(max_concurrency=4)

    def run(entry: _ScatterEntry) -> None:
        # Introduce artificial jitter so threads finish out of submission order.
        time.sleep(0.01 * (4 - entry.index))
        entry.deposit(entry.value * 10)

    gate.scatter([1, 2, 3, 4], run)
    assert [e.result for e in gate.gather()] == [10, 20, 30, 40]


def test_scatter_gate_exception_propagates():
    gate = ScatterGate(max_concurrency=2)
    sentinel = ValueError("worker boom")

    def run(entry: _ScatterEntry) -> None:
        if entry.index == 1:
            entry.deposit_exception(sentinel)
        else:
            time.sleep(0.02)
            entry.deposit(entry.value)

    gate.scatter([1, 2], run)
    with pytest.raises(ValueError, match="worker boom"):
        gate.gather()


def test_scatter_gate_all_workers_finish_before_reraise():
    """gather() should wait for ALL workers before re-raising."""
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
    with pytest.raises(RuntimeError):
        gate.gather()
    # All three workers deposited before gather returned.
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


def test_pipeline_scatter_gather_worker_exception_propagates():
    class _Boom:
        def __call__(self, value: int) -> int:
            if value == 3:
                raise ValueError("bad item")
            return value

    pipeline = Pipeline([Scatter(max_concurrency=4), _Boom(), Gather()])
    with pytest.raises(ValueError, match="bad item"):
        pipeline([1, 2, 3, 4])


def test_pipeline_scatter_gather_type_contract():
    contract = Pipeline([Scatter(max_concurrency=1), _Double(), Gather()]).validate()
    assert contract is not None


# ---------------------------------------------------------------------------
# Validation: structure
# ---------------------------------------------------------------------------

def test_validate_unmatched_scatter():
    with pytest.raises(PipelineValidationError, match="no matching Gather"):
        Pipeline([Scatter(max_concurrency=1), _Double()]).validate()


def test_validate_unmatched_gather():
    with pytest.raises(PipelineValidationError, match="no matching opener"):
        Pipeline([_Double(), Gather()]).validate()


def test_validate_nested_scatter_forbidden():
    with pytest.raises(PipelineValidationError, match="Nested Scatter"):
        Pipeline([
            Scatter(max_concurrency=1),
            Scatter(max_concurrency=1),
            _Double(),
            Gather(),
            Gather(),
        ]).validate()


def test_validate_batch_inside_scatter_allowed():
    from ml_pipes import Batch, UnBatch

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
    from ml_pipes import Batch

    with pytest.raises(PipelineValidationError, match="interleave"):
        Pipeline([
            Scatter(max_concurrency=1),
            Batch(size=1),
            _Double(),
            Gather(),   # closes Scatter but Batch is still open
        ]).validate()


def test_validate_interleaved_batch_scatter_forbidden():
    """Batch → Scatter → UnBatch (no Gather before UnBatch) must be rejected."""
    from ml_pipes import Batch, UnBatch

    with pytest.raises(PipelineValidationError, match="interleave"):
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
