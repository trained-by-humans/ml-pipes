from __future__ import annotations

from ..tracing import InvocationTrace
from .serial_collector import SerialCollector


class PrintCollector(SerialCollector):
    """Prints each trace to stdout. Useful for development and debugging.

    The last received trace is kept in ``last_trace`` and can be reprinted
    at any time via ``print_trace()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_trace: InvocationTrace | None = None

    def _collect(self, trace: InvocationTrace) -> None:
        self.last_trace = trace
        self.print_trace(trace)

    def print_trace(self, trace: InvocationTrace | None = None, indent: int = 0) -> None:
        """Print *trace* (defaults to ``last_trace``) to stdout."""
        if trace is None:
            trace = self.last_trace
        if trace is None:
            return
        prefix = "  " * indent
        fracs = trace.span_fractions()
        for span in trace.spans:
            mark = " !" if span.error else ""
            print(
                f"{prefix}  {span.label:30s} {span.duration_s * 1000:7.2f}ms"
                f"  ({fracs[span.label] * 100:5.1f}%){mark}"
            )
            if span.child_trace is not None:
                ct = span.child_trace
                if ct.scatter_workers is not None:
                    annotation = f" [n_items={ct.batch_size}, concurrency={ct.scatter_workers}]"
                elif ct.batch_size is not None:
                    annotation = f" [batch_size={ct.batch_size}]"
                else:
                    annotation = ""
                print(f"{prefix}    ↳ child trace{annotation}:")
                self.print_trace(span.child_trace, indent + 2)
        print(f"{prefix}  {'total':30s} {trace.total_duration_s * 1000:7.2f}ms")
