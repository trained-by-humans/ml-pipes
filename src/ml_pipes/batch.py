from __future__ import annotations

import threading
from typing import Any


# Sentinel placed in an entry's result slot to signal that it has been promoted
# to batch leader by the timeout path.
_LEADER = object()


class _Entry:
    __slots__ = ("value", "event", "result")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.event: threading.Event = threading.Event()
        self.result: list = []  # [value] | [_LEADER, batch] | [BaseException]


class _BatchGate:
    """
    Coordination primitive for the Batch/UnBatch operator pair.

    Multiple threads call enter() concurrently.  When enough samples have
    accumulated (up to *size*) or *timeout* seconds have elapsed since the
    first sample arrived, one thread is elected leader.

    Leader path:  enter() returns (list_of_inputs, True).
                  The pipeline runs the batch region and calls distribute().
    Waiter path:  enter() blocks until distribute() fires the thread's event,
                  then returns (per_sample_result, False).

    Exception path: if the batch region raises, the leader calls
                    distribute_exception() before re-raising so that all
                    waiting threads receive the exception instead of blocking
                    forever.
    """

    def __init__(self, size: int, timeout: float) -> None:
        self._size = size
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pending: list[_Entry] = []
        self._timer: threading.Timer | None = None
        # Per-leader-thread state so concurrent batches don't interfere.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enter(self, value: Any) -> tuple[Any, bool]:
        """
        Register *value* for the next batch.

        Returns ``(list_of_inputs, True)``  for the leader thread.
        Returns ``(per_sample_result, False)`` for waiter threads (blocks).
        """
        entry = _Entry(value)

        with self._lock:
            idx = len(self._pending)
            self._pending.append(entry)

            if len(self._pending) == self._size:
                batch = self._take_pending()
                self._local.batch = batch
                self._local.leader_idx = idx  # = size - 1
                return [e.value for e in batch], True

            if idx == 0:
                # First arrival — start the timeout clock.
                self._timer = threading.Timer(self._timeout, self._on_timeout)
                self._timer.daemon = True
                self._timer.start()

        # ---- waiter path ----
        entry.event.wait()

        if entry.result[0] is _LEADER:
            # Timeout promoted this thread to leader.
            batch = entry.result[1]
            self._local.batch = batch
            self._local.leader_idx = 0
            return [e.value for e in batch], True

        if isinstance(entry.result[0], BaseException):
            raise entry.result[0]

        return entry.result[0], False

    def distribute(self, results: list[Any]) -> Any:
        """
        Called by the pipeline (leader thread) at the UnBatch position.

        Writes each result to its waiter's slot and fires their events.
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
            entry.result.append(result)
            entry.event.set()

        return results[leader_idx]

    def distribute_exception(self, exc: BaseException) -> None:
        """
        Propagate *exc* to all waiters so they unblock and raise instead of
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
            entry.result.append(exc)
            entry.event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _take_pending(self) -> list[_Entry]:
        """Drain _pending and cancel any running timer.  Must hold self._lock."""
        batch = self._pending[:]
        self._pending.clear()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        return batch

    def _on_timeout(self) -> None:
        """Timer callback: promote the oldest waiter to leader."""
        with self._lock:
            if not self._pending:
                # Race: batch was already collected by a full-batch leader.
                return
            batch = self._take_pending()

        # Promote the first waiter.  Its event.wait() will return and it will
        # check for the _LEADER sentinel in its result slot.
        leader_entry = batch[0]
        leader_entry.result.extend([_LEADER, batch])
        leader_entry.event.set()
