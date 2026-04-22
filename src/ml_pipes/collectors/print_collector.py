from __future__ import annotations

from ..tracing import InvocationTrace, TraceCollector


class PrintCollector(TraceCollector):
    """Prints each trace to stdout. Useful for development and debugging."""

    def on_trace(self, trace: InvocationTrace) -> None:
        self._print_trace(trace, indent=0)

    def _print_trace(self, trace: InvocationTrace, indent: int) -> None:
        prefix = "  " * indent
        fracs = trace.span_fractions()
        for span in trace.spans:
            mark = " !" if span.error else ""
            print(
                f"{prefix}  {span.label:30s} {span.duration_s * 1000:7.2f}ms"
                f"  ({fracs[span.label] * 100:5.1f}%){mark}"
            )
            if span.child_trace is not None:
                bs = (
                    f" [batch_size={span.child_trace.batch_size}]"
                    if span.child_trace.batch_size is not None
                    else ""
                )
                print(f"{prefix}    ↳ child trace{bs}:")
                self._print_trace(span.child_trace, indent + 2)
        print(f"{prefix}  {'total':30s} {trace.total_duration_s * 1000:7.2f}ms")
