from __future__ import annotations

from threading import Lock

from ml_pipes.inspection.views import StepFormatter, ValueFormatter

_LOCK = Lock()
_VALUE_FORMATTERS: dict[type, ValueFormatter] = {}
_STEP_FORMATTERS: dict[type, StepFormatter] = {}


def value_formatters() -> dict[type, ValueFormatter]:
    with _LOCK:
        return dict(_VALUE_FORMATTERS)


def register_value_formatter(value_type: type, formatter: ValueFormatter) -> None:
    with _LOCK:
        _VALUE_FORMATTERS[value_type] = formatter


def step_formatters() -> dict[type, StepFormatter]:
    with _LOCK:
        return dict(_STEP_FORMATTERS)


def register_step_formatter(operator_type: type, formatter: StepFormatter) -> None:
    with _LOCK:
        _STEP_FORMATTERS[operator_type] = formatter
