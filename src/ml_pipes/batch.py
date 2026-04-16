from __future__ import annotations

import threading
from typing import Any


class _Entry:
    __slots__ = ("value", "event", "result", "exception")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.event: threading.Event = threading.Event()  # fired exactly once by distribute / distribute_exception
        self.result: Any = None                          # set by distribute()
        self.exception: BaseException | None = None      # set by distribute_exception()


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
    This thread's per-sample result is ready to consume.
    """
    __slots__ = ("result",)

    def __init__(self, result: Any) -> None:
        self.result = result


class BatchGate:
    """
    Coordination primitive for the Batch/UnBatch operator pair.

    All threads enter a lobby and wait on _lobby_cond. When enough samples
    accumulate (up to *size*) or *timeout* seconds elapse, all threads are
    woken together. One wins the race to drain _pending and becomes the
    leader; the rest become followers and wait on their per-entry event for
    their result.

    _lobby_cond is shared across all threads because only one lobby is active
    at a time. Per-entry events are used for result delivery so that concurrent
    batches cannot trigger each other's followers.

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
        self._lobby_cond = threading.Condition()
        self._pending: list[_Entry] = []
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
        with self._lobby_cond:
            self._pending.append(entry)

            if len(self._pending) == self._size:
                # Full batch — wake all lobby waiters.
                self._lobby_cond.notify_all()
            else:
                # Partial batch — wait for either a full batch or timeout.
                self._lobby_cond.wait(timeout=self._timeout)
                # Cascade: wake any remaining lobby waiters so everyone races together.
                self._lobby_cond.notify_all()

            # Race to drain _pending — only the thread whose entry is still
            # in _pending wins.  A woken loser whose entry was already collected
            # by another leader must not drain entries belonging to the next batch.
            if entry in self._pending:
                batch = self._pending[:]
                self._pending.clear()
                self._local.batch = batch
                self._local.leader_idx = batch.index(entry)
                return LeaderBatch([e.value for e in batch])

        # Phase 2 — batch operation: wait for the leader to fire our entry's event.
        entry.event.wait()

        if entry.exception is not None:
            raise entry.exception
        return FollowerResult(entry.result)

    def distribute(self, results: list[Any]) -> Any:
        """
        Called by the pipeline (leader thread) at the UnBatch position.

        Writes each follower's result and fires their event.
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
            entry.event.set()

        return results[leader_idx]

    def distribute_exception(self, exc: BaseException) -> None:
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
            entry.event.set()
