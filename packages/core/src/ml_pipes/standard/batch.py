from __future__ import annotations

import threading
import time
from typing import Any, TypeVar, get_args, get_origin

from ml_pipes.operator import Operator
from ml_pipes.region import RegionCloser, RegionExecutor, RegionOpener, RegionTraceLike
from ml_pipes.tracing import InvocationTrace, StepSpan, TracingConfig, _NoOpTrace


class _Entry:
    __slots__ = ("value", "event", "result", "exception", "batch_span")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.event: threading.Event = threading.Event()
        self.result: Any = None
        self.exception: BaseException | None = None
        self.batch_span: Any = None


class LeaderBatch:
    """Returned by BatchGate.enter() for the thread elected as batch leader.

    The leader receives the raw per-sample inputs and is responsible for
    running the batch region and calling distribute().
    """

    __slots__ = ("inputs",)

    def __init__(self, inputs: list[Any]) -> None:
        self.inputs = inputs


class FollowerResult:
    """Returned by BatchGate.enter() for a waiter thread.

    The leader has already run the batch region and distributed results.
    This thread's per-sample result is ready to consume. If the leader
    failed, exception is set and result is None.
    """

    __slots__ = ("result", "batch_span", "exception")

    def __init__(
        self,
        result: Any,
        batch_span: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        self.result = result
        self.batch_span = batch_span
        self.exception = exception


class BatchGate:
    """
    Coordination primitive for the Batch/UnBatch operator pair.

    All threads enter a lobby and wait on a per-batch Condition. When enough
    samples accumulate (up to *size*) or *timeout* seconds elapse, all threads
    in that batch are woken together. One wins the race to drain _pending and
    becomes the leader; the rest become followers and wait on their per-entry
    event for their result.

    A new Condition is created for each batch (by the first arriving thread),
    so notify_all() cannot bleed across batch generations and wake threads
    that belong to the next batch.

    Leader path:  enter() returns LeaderBatch.
                  The pipeline runs the batch region and calls distribute().
    Follower path: enter() blocks on entry.event until the leader fires it,
                  then returns FollowerResult.

    Exception path: if the batch region raises, the leader calls
                    distribute_exception() before re-raising so that all
                    followers receive the exception instead of blocking
                    forever.
    """

    def __init__(self, size: int, timeout: float) -> None:
        self._size = size
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pending: list[_Entry] = []
        self._batch_cond: threading.Condition | None = None
        self._local = threading.local()

    def enter(self, value: Any) -> LeaderBatch | FollowerResult:
        """
        Register *value* for the next batch.

        Returns ``LeaderBatch`` for the thread that wins the drain race.
        Returns ``FollowerResult`` for all other threads.
        Raises the batch-region exception for followers when the leader fails.
        """

        entry = _Entry(value)

        with self._lock:
            if not self._pending:
                self._batch_cond = threading.Condition(self._lock)

            cond = self._batch_cond
            self._pending.append(entry)

            if len(self._pending) == self._size:
                cond.notify_all()
            else:
                cond.wait(timeout=self._timeout)
                cond.notify_all()

            if entry in self._pending:
                batch = self._pending[:]
                self._pending.clear()
                self._batch_cond = None
                self._local.batch = batch
                self._local.leader_idx = batch.index(entry)
                return LeaderBatch([item.value for item in batch])

        entry.event.wait()
        return FollowerResult(entry.result, batch_span=entry.batch_span, exception=entry.exception)

    def distribute(self, results: list[Any], batch_span: Any = None) -> Any:
        """Write follower results and return the leader's own result."""

        batch: list[_Entry] = self._local.batch
        leader_idx: int = self._local.leader_idx

        if len(results) != len(batch):
            raise RuntimeError(
                f"Batch size mismatch: {len(batch)} inputs but {len(results)} results"
            )

        del self._local.batch, self._local.leader_idx

        for index, (entry, result) in enumerate(zip(batch, results)):
            if index == leader_idx:
                continue
            entry.result = result
            entry.batch_span = batch_span
            entry.event.set()

        return results[leader_idx]

    def distribute_exception(self, exc: BaseException, batch_span: Any = None) -> None:
        """Propagate *exc* to all followers so they unblock and raise."""

        batch: list[_Entry] | None = getattr(self._local, "batch", None)
        if batch is None:
            return

        leader_idx: int | None = getattr(self._local, "leader_idx", None)

        if hasattr(self._local, "batch"):
            del self._local.batch
        if hasattr(self._local, "leader_idx"):
            del self._local.leader_idx

        for index, entry in enumerate(batch):
            if index == leader_idx:
                continue
            entry.exception = exc
            entry.batch_span = batch_span
            entry.event.set()


BatchItemT = TypeVar("BatchItemT")


@Operator
class UnBatch(RegionCloser[list[BatchItemT], BatchItemT]):
    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        if current_output is not None and get_origin(current_output) is list:
            args = get_args(current_output)
            return (Any,), args[0] if args else Any
        return (Any,), Any


@Operator
class Batch(RegionOpener[BatchItemT, list[BatchItemT]]):
    closing_type = UnBatch

    def __init__(self, size: int, timeout: float = 0.05) -> None:
        self.gate = BatchGate(size, timeout)

    def run_region(
        self,
        current: BatchItemT,
        label: str,
        execute_region: RegionExecutor[list[BatchItemT], Any],
        trace: RegionTraceLike,
        cfg: TracingConfig | None,
    ) -> Any:
        del cfg
        gate = self.gate

        t_gate_enter = time.perf_counter()
        outcome = gate.enter(current)
        gate_blocked_duration = time.perf_counter() - t_gate_enter

        if not isinstance(outcome, LeaderBatch):
            batch_region_duration = outcome.batch_span.duration_s if outcome.batch_span is not None else 0.0
            lobby_wait_duration = gate_blocked_duration - batch_region_duration
            trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, lobby_wait_duration))
            if outcome.batch_span is not None:
                trace.spans.append(outcome.batch_span)
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.result

        trace.spans.append(StepSpan(f"{label}[wait]", t_gate_enter, gate_blocked_duration))
        current = outcome.inputs
        batch_size = len(current) if hasattr(current, "__len__") else None
        collecting = isinstance(trace, InvocationTrace)
        child_trace = InvocationTrace(batch_size=batch_size) if collecting else _NoOpTrace(batch_size=batch_size)

        t_region = time.perf_counter()
        try:
            current, child_trace = execute_region(current, child_trace)
        except Exception as exc:
            error_span = StepSpan(
                label,
                t_region,
                child_trace.total_duration_s,
                error=True,
                child_trace=child_trace if collecting else None,
                operator_type=type(self),
            )
            trace.spans.append(error_span)
            gate.distribute_exception(exc, batch_span=error_span if collecting else None)
            raise

        batch_span = StepSpan(
            label,
            t_region,
            child_trace.total_duration_s,
            child_trace=child_trace if collecting else None,
            operator_type=type(self),
        )
        trace.spans.append(batch_span)
        return gate.distribute(current, batch_span=batch_span if collecting else None)

    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        del stored_annotations, expand_output_annotation, validation_error_type
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out
