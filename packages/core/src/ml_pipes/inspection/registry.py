from __future__ import annotations

from threading import Lock
from typing import Any, Callable, TypeVar, cast

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


class FormatterRegistry:
    def __init__(self, parent: FormatterRegistry | None = None) -> None:
        self._parent = parent
        self._lock = Lock()
        self._value_formatters: dict[type[Any], AnyValueFormatter] = {}
        self._step_formatters: dict[type[Any], StepFormatter] = {}

    def value_formatters(self) -> dict[type[Any], AnyValueFormatter]:
        with self._lock:
            return dict(self._value_formatters)

    def step_formatters(self) -> dict[type[Any], StepFormatter]:
        with self._lock:
            return dict(self._step_formatters)

    def register_value_formatter(
        self,
        value_type: type[ValueT],
        formatter: ValueFormatter[ValueT],
    ) -> None:
        with self._lock:
            self._value_formatters[value_type] = cast(AnyValueFormatter, formatter)

    def register_step_formatter(self, operator_type: type[Any], formatter: StepFormatter) -> None:
        with self._lock:
            self._step_formatters[operator_type] = formatter

    def get_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        with self._lock:
            return self._value_formatters.get(value_type)

    def get_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        with self._lock:
            return self._step_formatters.get(operator_type)

    def find_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        formatter = self.get_value_formatter(value_type)
        if formatter is not None:
            return formatter

        if self._parent is not None:
            formatter = self._parent._find_exact_value_formatter(value_type)
            if formatter is not None:
                return formatter

        for registered_type, formatter in self.value_formatters().items():
            if issubclass(value_type, registered_type):
                return formatter

        if self._parent is not None:
            return self._parent._find_subclass_value_formatter(value_type)
        return None

    def find_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        formatter = self.get_step_formatter(operator_type)
        if formatter is not None:
            return formatter

        if self._parent is not None:
            formatter = self._parent._find_exact_step_formatter(operator_type)
            if formatter is not None:
                return formatter

        for registered_type, formatter in self.step_formatters().items():
            if issubclass(operator_type, registered_type):
                return formatter

        if self._parent is not None:
            return self._parent._find_subclass_step_formatter(operator_type)
        return None

    def _find_exact_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        formatter = self.get_value_formatter(value_type)
        if formatter is not None:
            return formatter
        if self._parent is not None:
            return self._parent._find_exact_value_formatter(value_type)
        return None

    def _find_subclass_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        for registered_type, formatter in self.value_formatters().items():
            if issubclass(value_type, registered_type):
                return formatter
        if self._parent is not None:
            return self._parent._find_subclass_value_formatter(value_type)
        return None

    def _find_exact_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        formatter = self.get_step_formatter(operator_type)
        if formatter is not None:
            return formatter
        if self._parent is not None:
            return self._parent._find_exact_step_formatter(operator_type)
        return None

    def _find_subclass_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        for registered_type, formatter in self.step_formatters().items():
            if issubclass(operator_type, registered_type):
                return formatter
        if self._parent is not None:
            return self._parent._find_subclass_step_formatter(operator_type)
        return None
