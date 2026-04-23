from __future__ import annotations

import threading
from abc import abstractmethod

from ..tracing import InvocationTrace, TraceCollector


class SerialCollector(TraceCollector):
    """Base for collectors that must not be called concurrently.

    on_trace() is protected by a lock so concurrent pipeline threads are
    serialised before reaching _collect(). Subclasses implement _collect().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def on_trace(self, trace: InvocationTrace) -> None:
        with self._lock:
            self._collect(trace)

    @abstractmethod
    def _collect(self, trace: InvocationTrace) -> None: ...
