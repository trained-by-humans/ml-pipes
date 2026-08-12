from __future__ import annotations

from threading import Lock

from ml_pipes.inspection.views import OutputFormatter, SpanFormatter

_LOCK = Lock()
_OUTPUT_FORMATTERS: dict[type, OutputFormatter] = {}
_SPAN_FORMATTERS: dict[type, SpanFormatter] = {}


def output_formatters() -> dict[type, OutputFormatter]:
    with _LOCK:
        return dict(_OUTPUT_FORMATTERS)


def register_output_formatter(type_: type, formatter: OutputFormatter) -> None:
    with _LOCK:
        _OUTPUT_FORMATTERS[type_] = formatter


def span_formatters() -> dict[type, SpanFormatter]:
    with _LOCK:
        return dict(_SPAN_FORMATTERS)


def register_span_formatter(operator_type: type, formatter: SpanFormatter) -> None:
    with _LOCK:
        _SPAN_FORMATTERS[operator_type] = formatter
