import threading

import numpy as np
import pytest

from ml_pipes import (
    Batch,
    Collate,
    Distribute,
    Pipeline,
    PipelineValidationError,
    UnBatch,
)
from ml_pipes.types import RuntimeOutputs, TensorPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DoubleBatchOp:
    """Doubles every value in the batch list."""
    def __call__(self, values: list) -> list:
        return [v * 2 for v in values]


class _IdentityBatchOp:
    """Returns the batch list unchanged."""
    def __call__(self, values: list) -> list:
        return values


class _FailBatchOp:
    """Always raises from inside the batch region."""
    def __call__(self, values: list) -> list:
        raise RuntimeError("batch failed")


def _make_pipeline(size: int, timeout: float, batch_op) -> Pipeline:
    return Pipeline([Batch(size=size, timeout=timeout), batch_op, UnBatch()])


def _run_threads(pipeline: Pipeline, inputs: list, timeout: float = 2.0) -> list:
    """
    Call pipeline(inputs[i]) from N threads started simultaneously via a
    Barrier.  Returns a list where results[i] is the return value of thread i
    (or the exception it raised).
    """
    n = len(inputs)
    results: list = [None] * n
    barrier = threading.Barrier(n)

    def call(i: int) -> None:
        barrier.wait()
        try:
            results[i] = pipeline(inputs[i])
        except Exception as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)

    return results


# ---------------------------------------------------------------------------
# Full batch coordination
# ---------------------------------------------------------------------------

def test_full_batch_each_thread_gets_its_own_result_doubled():
    # 3 threads form one batch of 3.  Each value is doubled.
    # Because _DoubleBatchOp maps value→value*2 positionally and routing
    # preserves per-thread identity, results[i] == inputs[i] * 2.
    pipeline = _make_pipeline(size=3, timeout=0.5, batch_op=_DoubleBatchOp())
    results = _run_threads(pipeline, [0, 1, 2])

    assert results[0] == 0
    assert results[1] == 2
    assert results[2] == 4


def test_full_batch_result_routing_no_mixing():
    # Identity batch region: each thread must get its own input back.
    pipeline = _make_pipeline(size=3, timeout=0.5, batch_op=_IdentityBatchOp())
    results = _run_threads(pipeline, [10, 20, 30])

    assert results[0] == 10
    assert results[1] == 20
    assert results[2] == 30


def test_concurrent_batches_do_not_interfere():
    # 8 threads → 2 batches of 4 run concurrently; no result cross-talk.
    pipeline = _make_pipeline(size=4, timeout=0.5, batch_op=_DoubleBatchOp())
    inputs = list(range(8))
    results = _run_threads(pipeline, inputs, timeout=5.0)

    for i, result in enumerate(results):
        assert result == i * 2, f"Thread {i}: expected {i * 2}, got {result}"


# ---------------------------------------------------------------------------
# Timeout / partial batch
# ---------------------------------------------------------------------------

def test_single_request_runs_after_timeout():
    pipeline = _make_pipeline(size=4, timeout=0.05, batch_op=_DoubleBatchOp())
    result = pipeline(7)
    assert result == 14


def test_partial_batch_runs_after_timeout():
    # 2 threads arrive, size=4 — timeout fires with a batch of 2.
    pipeline = _make_pipeline(size=4, timeout=0.1, batch_op=_DoubleBatchOp())
    results = _run_threads(pipeline, [3, 5], timeout=2.0)

    assert results[0] == 6
    assert results[1] == 10


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

def test_exception_in_batch_region_propagates_to_all_waiters():
    pipeline = _make_pipeline(size=3, timeout=0.5, batch_op=_FailBatchOp())
    results = _run_threads(pipeline, [0, 1, 2])

    assert all(isinstance(r, RuntimeError) for r in results), results
    assert all("batch failed" in str(r) for r in results)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def test_collate_concatenates_nchw_tensors_along_batch_dim():
    tensors = [
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
        TensorPayload(array=np.zeros((1, 3, 8, 8), dtype=np.float32), layout="NCHW", dtype="float32"),
    ]
    result = Collate()(tensors)
    assert result.array.shape == (3, 3, 8, 8)
    assert result.layout == "NCHW"
    assert result.dtype == "float32"


def test_collate_stacks_chw_tensors_adding_batch_dim():
    tensors = [
        TensorPayload(array=np.zeros((3, 8, 8), dtype=np.float32), layout="CHW", dtype="float32"),
        TensorPayload(array=np.zeros((3, 8, 8), dtype=np.float32), layout="CHW", dtype="float32"),
    ]
    result = Collate()(tensors)
    assert result.array.shape == (2, 3, 8, 8)


def test_collate_raises_on_empty_list():
    with pytest.raises(ValueError, match="empty"):
        Collate()([])


# ---------------------------------------------------------------------------
# Distribute
# ---------------------------------------------------------------------------

def test_distribute_splits_batch_dim_into_per_sample_outputs():
    batched = np.arange(12, dtype=np.float32).reshape(3, 4)
    outputs = RuntimeOutputs(
        tensors=(TensorPayload(array=batched, layout="UNKNOWN", dtype="float32"),),
        names=("preds",),
    )
    result = Distribute()(outputs)

    assert len(result) == 3
    for i, sample in enumerate(result):
        assert sample.tensors[0].array.shape == (1, 4)
        assert np.array_equal(sample.tensors[0].array, batched[i : i + 1])
        assert sample.names == ("preds",)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_raises_on_unmatched_batch():
    pipeline = Pipeline([Batch(size=2)])
    with pytest.raises(PipelineValidationError, match="no matching UnBatch"):
        pipeline.validate()


def test_validate_raises_on_unmatched_unbatch():
    pipeline = Pipeline([UnBatch()])
    with pytest.raises(PipelineValidationError, match="no matching Batch"):
        pipeline.validate()


def test_validate_raises_on_nested_batch():
    pipeline = Pipeline([Batch(size=4), Batch(size=2), UnBatch(), UnBatch()])
    with pytest.raises(PipelineValidationError, match="Nested Batch"):
        pipeline.validate()


def test_validate_accepts_matched_batch_unbatch_pair():
    pipeline = Pipeline([Batch(size=2), UnBatch()])
    pipeline.validate()  # must not raise


def test_validate_raises_on_context_op_inside_batch_region():
    from ml_pipes.context import Store
    pipeline = Pipeline([Batch(size=2), Store("x"), UnBatch()])
    with pytest.raises(PipelineValidationError, match="inside a Batch region"):
        pipeline.validate()
