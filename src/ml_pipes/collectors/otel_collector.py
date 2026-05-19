from __future__ import annotations

# Optional OpenTelemetry bridge.
# Install with: pip install ml-pipes[otel]
# Never imported by core.py or __init__.py — no implicit dependency.
from opentelemetry import trace as otel_trace

from .concurrent_collector import ConcurrentCollector
from ..tracing import InvocationTrace


class OtelCollector(ConcurrentCollector):
    """Bridges ml-pipes InvocationTraces to OpenTelemetry spans.

    Wire up a TracerProvider before use::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry import trace
        trace.set_tracer_provider(TracerProvider())
    """

    def __init__(self, tracer_name: str = "ml-pipes") -> None:
        super().__init__()
        self._tracer = otel_trace.get_tracer(tracer_name)

    def _collect(self, trace: InvocationTrace) -> None:
        with self._tracer.start_as_current_span("pipeline") as root:
            root.set_attribute("total_duration_ms", trace.total_duration_s * 1000)
            self._emit_spans(trace)

    def _emit_spans(self, trace: InvocationTrace) -> None:
        for span in trace.spans:
            with self._tracer.start_as_current_span(span.label) as s:
                s.set_attribute("duration_ms", span.duration_s * 1000)
                if span.error:
                    # Status(StatusCode) form required since opentelemetry-api 1.1;
                    # pyproject.toml pins >=1.20 so this is always safe.
                    s.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR))
                if span.input_shape is not None:
                    s.set_attribute("input_shape", str(span.input_shape))
                if span.output_shape is not None:
                    s.set_attribute("output_shape", str(span.output_shape))
                if span.child_trace is not None:
                    if span.child_trace.batch_size is not None:
                        s.set_attribute("batch_size", span.child_trace.batch_size)
                    self._emit_spans(span.child_trace)
