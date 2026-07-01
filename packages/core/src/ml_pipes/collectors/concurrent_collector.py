from __future__ import annotations

import logging
import queue
import threading
from abc import abstractmethod
from typing import Any

from ..tracing import InvocationTrace, TraceCollector

_SENTINEL = object()
_log = logging.getLogger(__name__)


class ConcurrentCollector(TraceCollector):
    """Base for collectors that process traces on a dedicated background thread.

    on_trace() enqueues the trace and returns immediately; the worker thread
    drains the queue and calls _collect() serially off the hot path.
    Subclasses implement _collect().

    Lifecycle::

        with MyCollector() as collector:
            pipeline.set_tracing(collector)
            ...

    Or manually::

        collector = MyCollector()
        pipeline.set_tracing(collector)
        ...
        collector.flush()   # wait for all queued traces to be processed
        collector.stop()    # shut down the worker thread
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._stopped = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def on_trace(self, trace: InvocationTrace) -> None:
        if self._stopped:
            _log.warning("ConcurrentCollector is stopped; trace dropped")
            return
        self._queue.put(trace)

    def flush(self) -> None:
        """Block until all currently queued traces have been processed."""
        self._queue.join()

    def stop(self) -> None:
        """Flush and shut down the worker thread. Safe to call more than once."""
        if self._worker.is_alive():
            self._stopped = True
            self._queue.put(_SENTINEL)
            self._worker.join()

    def __enter__(self) -> ConcurrentCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @abstractmethod
    def _collect(self, trace: InvocationTrace) -> None: ...

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                self._collect(item)
            finally:
                self._queue.task_done()
