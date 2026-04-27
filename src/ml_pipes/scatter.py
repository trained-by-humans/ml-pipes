from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class _ScatterEntry:
    __slots__ = ("value", "index", "event", "result", "child_trace", "exception")

    def __init__(self, value: Any, index: int) -> None:
        self.value = value
        self.index = index
        self.event: threading.Event = threading.Event()
        self.result: Any = None
        self.child_trace: Any = None  # InvocationTrace | None, written before event.set()
        self.exception: BaseException | None = None

    def deposit(self, result: Any, child_trace: Any = None) -> None:
        self.result = result
        self.child_trace = child_trace  # written before event.set() — happens-before guarantee
        self.event.set()

    def deposit_exception(self, exc: BaseException, child_trace: Any = None) -> None:
        self.exception = exc
        self.child_trace = child_trace  # written before event.set() — happens-before guarantee
        self.event.set()


class ScatterGate:
    """
    Coordination primitive for the Scatter/Gather operator pair.

    One caller fans out a list of items to N worker threads; each worker runs
    the scatter region independently with a fresh Context, then deposits its
    result.  The original thread blocks at gather() until all workers have
    deposited, then collects results in submission order.

    Exception handling: if any worker raises, its exception is captured in its
    entry.  gather() re-raises the first exception seen after all workers finish
    (others are allowed to complete — no cancellation).

    Scatter path:  scatter(items) → submits one task per item to the executor.
    Gather path:   gather()       → blocks until all entries are set, then returns
                                    list[_ScatterEntry] in submission order (or raises).
    """

    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max_concurrency
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self._local = threading.local()

    def scatter(self, items: list[Any], run_region: Any) -> None:
        entries = [_ScatterEntry(item, i) for i, item in enumerate(items)]
        self._local.entries = entries
        for entry in entries:
            self._executor.submit(run_region, entry)

    def gather(self) -> list[_ScatterEntry]:
        entries: list[_ScatterEntry] = self._local.entries
        del self._local.entries

        first_exc: BaseException | None = None
        for entry in entries:
            entry.event.wait()
            if entry.exception is not None and first_exc is None:
                first_exc = entry.exception

        if first_exc is not None:
            raise first_exc

        return entries
