from __future__ import annotations

from importlib import import_module

from ml_pipes.collectors.concurrent_collector import ConcurrentCollector
from ml_pipes.tracing import InvocationTrace


def _load_otel_trace() -> object:
    try:
        return import_module("opentelemetry.trace")
    except ImportError as exc:  # pragma: no cover - depends on optional dependency state
        raise ImportError(
            "ml_pipes.collectors.OtelCollector requires the optional otel extra. "
            "Install it with `pip install ml-pipes[otel]`."
        ) from exc


class OtelCollector(ConcurrentCollector):
    """Bridges ml-pipes InvocationTraces to OpenTelemetry spans.

    Wire up a TracerProvider before use::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry import trace
        trace.set_tracer_provider(TracerProvider())
    """

    def __init__(self, tracer_name: str = "ml-pipes") -> None:
        super().__init__()
        otel_trace = _load_otel_trace()
        self._tracer = otel_trace.get_tracer(tracer_name)
        self._otel_trace = otel_trace

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
                    s.set_status(self._otel_trace.Status(self._otel_trace.StatusCode.ERROR))
                if span.child_trace is not None:
                    if span.child_trace.batch_size is not None:
                        s.set_attribute("batch_size", span.child_trace.batch_size)
                    self._emit_spans(span.child_trace)
