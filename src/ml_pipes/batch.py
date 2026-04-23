from __future__ import annotations

import threading
from typing import Any


class _Entry:
    __slots__ = ("value", "event", "result", "exception", "batch_span")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.event: threading.Event = threading.Event()  # fired exactly once by distribute / distribute_exception
        self.result: Any = None                          # set by distribute()
        self.exception: BaseException | None = None      # set by distribute_exception()
        self.batch_span: Any = None                      # StepSpan | None, written by leader before event.set()


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

    def __init__(self, result: Any, batch_span: Any = None, exception: BaseException | None = None) -> None:
        self.result = result
        self.batch_span = batch_span  # StepSpan | None, copied from leader via _Entry
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
        self._batch_cond: threading.Condition | None = None  # per-batch, recreated each cycle
        # Per-leader-thread state so concurrent batches don't interfere.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enter(self, value: Any) -> LeaderBatch | FollowerResult:
        """
        Register *value* for the next batch.

        Returns ``LeaderBatch``    for the thread that wins the drain race.
        Returns ``FollowerResult`` for all other threads (blocks until the
                                    leader distributes results).
        Raises the batch-region exception for followers when the leader fails.
        """
        entry = _Entry(value)

        # Phase 1 — batch formation: wait in the lobby for a full batch or timeout.
        with self._lock:
            if not self._pending:
                # First arrival for this batch — create a fresh condition so that
                # notify_all() cannot bleed into a concurrent or future batch.
                self._batch_cond = threading.Condition(self._lock)

            # Keep a local reference to this batch condition
            cond = self._batch_cond
            self._pending.append(entry)

            if len(self._pending) == self._size:
                # Full batch — wake all lobby waiters.
                cond.notify_all()
            else:
                # Partial batch — wait for either a full batch or timeout.
                cond.wait(timeout=self._timeout)
                # Cascade: wake any remaining batch members so everyone races together.
                cond.notify_all()

            # Race to drain _pending — only the thread whose entry is still
            # present wins.  A woken follower whose entry was already collected
            # must not drain entries that belong to the next batch.
            if entry in self._pending:
                batch = self._pending[:]
                self._pending.clear()
                self._batch_cond = None  # release the condition for this batch cycle
                self._local.batch = batch
                self._local.leader_idx = batch.index(entry)
                return LeaderBatch([e.value for e in batch])

        # Phase 2 — batch operation: wait for the leader to fire our entry's event.
        entry.event.wait()

        return FollowerResult(entry.result, batch_span=entry.batch_span, exception=entry.exception)

    def distribute(self, results: list[Any], batch_span: Any = None) -> Any:
        """
        Called by the pipeline (leader thread) at the UnBatch position.

        Writes each follower's result (and optional batch_span) and fires their event.
        Returns the leader's own result.
        """
        batch: list[_Entry] = self._local.batch
        leader_idx: int = self._local.leader_idx

        if len(results) != len(batch):
            # Do NOT delete _local state yet — distribute_exception will need it.
            raise RuntimeError(
                f"Batch size mismatch: {len(batch)} inputs but {len(results)} results"
            )

        # Clean up before signalling so re-entrant pipelines are safe.
        del self._local.batch, self._local.leader_idx

        for i, (entry, result) in enumerate(zip(batch, results)):
            if i == leader_idx:
                continue  # leader is not waiting — skip
            entry.result = result
            entry.batch_span = batch_span  # written before event.set() — happens-before guarantee
            entry.event.set()

        return results[leader_idx]

    def distribute_exception(self, exc: BaseException, batch_span: Any = None) -> None:
        """
        Propagate *exc* to all followers so they unblock and raise instead of
        hanging.  Safe to call even if distribute() already cleaned up.
        """
        batch: list[_Entry] | None = getattr(self._local, "batch", None)
        if batch is None:
            return

        leader_idx: int | None = getattr(self._local, "leader_idx", None)

        if hasattr(self._local, "batch"):
            del self._local.batch
        if hasattr(self._local, "leader_idx"):
            del self._local.leader_idx

        for i, entry in enumerate(batch):
            if i == leader_idx:
                continue
            entry.exception = exc
            entry.batch_span = batch_span  # written before event.set() — happens-before guarantee
            entry.event.set()

