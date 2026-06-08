from __future__ import annotations

from .serial_collector import SerialCollector
from ..tracing import InvocationTrace


class CaptureCollector(SerialCollector):
    """Capture the most recent trace for simple one-shot inspection use cases."""

    def __init__(self) -> None:
        super().__init__()
        self.last_trace: InvocationTrace | None = None

    def _collect(self, trace: InvocationTrace) -> None:
        self.last_trace = trace
