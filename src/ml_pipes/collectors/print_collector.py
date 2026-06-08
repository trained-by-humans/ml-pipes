from __future__ import annotations

from .capture_collector import CaptureCollector
from ..tracing import InvocationTrace


class PrintCollector(CaptureCollector):
    """Prints each trace to stdout. Useful for development and debugging.

    The last received trace is kept in ``last_trace`` and can be reprinted
    at any time via ``print_trace()``.
    """

    def __init__(self) -> None:
        super().__init__()

    def _collect(self, trace: InvocationTrace) -> None:
        super()._collect(trace)
        self.print_trace(trace)

    def print_trace(self, trace: InvocationTrace | None = None) -> None:
        """Print *trace* (defaults to ``last_trace``) to stdout."""
        if trace is None:
            trace = self.last_trace
        if trace is None:
            return
        print(trace)
