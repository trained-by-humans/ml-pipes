from __future__ import annotations

import queue
import threading
from typing import Any

from ..tracing import InvocationTrace, TraceCollector

_SENTINEL = object()


class AsyncCollector(TraceCollector):
    """Wraps any TraceCollector and dispatches on_trace calls to a background thread.

    The pipeline enqueues the trace and returns immediately; the worker drains
    the queue and forwards each trace to the inner collector off the hot path.

    Usage::

        with AsyncCollector(PrintCollector()) as collector:
            pipeline.set_tracing(collector)
            for frame in stream:
                pipeline(frame)

    Or manage lifecycle manually::

        collector = AsyncCollector(PrintCollector())
        pipeline.set_tracing(collector)
        ...
        collector.flush()   # wait for all queued traces to be processed
        collector.stop()    # shut down the worker thread
    """

    def __init__(self, inner: TraceCollector, maxsize: int = 0) -> None:
        self._inner = inner
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def on_trace(self, trace: InvocationTrace) -> None:
        self._queue.put(trace)

    def flush(self) -> None:
        """Block until all currently queued traces have been processed."""
        self._queue.join()

    def stop(self) -> None:
        """Flush and shut down the worker thread. Safe to call more than once."""
        if self._worker.is_alive():
            self._queue.put(_SENTINEL)
            self._worker.join()

    def __enter__(self) -> AsyncCollector:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                self._inner.on_trace(item)
            finally:
                self._queue.task_done()
