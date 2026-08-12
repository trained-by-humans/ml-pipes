from __future__ import annotations

from typing import Any, Callable, TypeVar, cast
from threading import Lock

import numpy as np

from ml_pipes.inspection.views import OutputBlock, StepView
from ml_pipes.tracing import StepSpan

ValueT = TypeVar("ValueT")
ValueFormatter = Callable[[ValueT], list[OutputBlock]]
StepFormatter = Callable[
    [StepSpan, np.ndarray | None],
    tuple[StepView, np.ndarray | None],
]
AnyValueFormatter = ValueFormatter[Any]

_LOCK = Lock()
_VALUE_FORMATTERS: dict[type[Any], AnyValueFormatter] = {}
_STEP_FORMATTERS: dict[type[Any], StepFormatter] = {}


def value_formatters() -> dict[type[Any], AnyValueFormatter]:
    with _LOCK:
        return dict(_VALUE_FORMATTERS)


def register_value_formatter(
    value_type: type[ValueT],
    formatter: ValueFormatter[ValueT],
) -> None:
    with _LOCK:
        _VALUE_FORMATTERS[value_type] = cast(AnyValueFormatter, formatter)


def step_formatters() -> dict[type[Any], StepFormatter]:
    with _LOCK:
        return dict(_STEP_FORMATTERS)


def register_step_formatter(operator_type: type[Any], formatter: StepFormatter) -> None:
    with _LOCK:
        _STEP_FORMATTERS[operator_type] = formatter
