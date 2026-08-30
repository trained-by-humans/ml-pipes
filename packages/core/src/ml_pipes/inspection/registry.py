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
FormatterT = TypeVar("FormatterT")


def _type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _best_subclass_formatter_match(
    requested_type: type[Any],
    formatters: dict[type[Any], FormatterT],
) -> tuple[int, FormatterT] | None:
    best: tuple[int, FormatterT] | None = None
    mro = requested_type.mro()
    fallback_depth = len(mro)

    for registered_type, formatter in formatters.items():
        if not issubclass(requested_type, registered_type):
            continue
        try:
            depth = mro.index(registered_type)
        except ValueError:
            depth = fallback_depth
        if best is None or depth < best[0]:
            best = (depth, formatter)

    return best


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
        *,
        override: bool = False,
    ) -> None:
        with self._lock:
            if value_type in self._value_formatters and not override:
                raise ValueError(
                    f"A value formatter is already registered for type {_type_name(value_type)!r}. "
                    "Set override=True to explicitly replace it."
                )
            self._value_formatters[value_type] = cast(AnyValueFormatter, formatter)

    def register_step_formatter(
        self,
        operator_type: type[Any],
        formatter: StepFormatter,
        *,
        override: bool = False,
    ) -> None:
        with self._lock:
            if operator_type in self._step_formatters and not override:
                raise ValueError(
                    f"A step formatter is already registered for operator type {_type_name(operator_type)!r}. "
                    "Set override=True to explicitly replace it."
                )
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

        match = self._find_subclass_value_formatter_match(value_type)
        return match[1] if match is not None else None

    def find_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        formatter = self.get_step_formatter(operator_type)
        if formatter is not None:
            return formatter

        if self._parent is not None:
            formatter = self._parent._find_exact_step_formatter(operator_type)
            if formatter is not None:
                return formatter

        match = self._find_subclass_step_formatter_match(operator_type)
        return match[1] if match is not None else None

    def _find_exact_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        formatter = self.get_value_formatter(value_type)
        if formatter is not None:
            return formatter
        if self._parent is not None:
            return self._parent._find_exact_value_formatter(value_type)
        return None

    def _find_subclass_value_formatter(self, value_type: type[Any]) -> AnyValueFormatter | None:
        match = self._find_subclass_value_formatter_match(value_type)
        return match[1] if match is not None else None

    def _find_exact_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        formatter = self.get_step_formatter(operator_type)
        if formatter is not None:
            return formatter
        if self._parent is not None:
            return self._parent._find_exact_step_formatter(operator_type)
        return None

    def _find_subclass_step_formatter(self, operator_type: type[Any]) -> StepFormatter | None:
        match = self._find_subclass_step_formatter_match(operator_type)
        return match[1] if match is not None else None

    def _find_subclass_value_formatter_match(
        self,
        value_type: type[Any],
    ) -> tuple[int, AnyValueFormatter] | None:
        local_match = _best_subclass_formatter_match(value_type, self.value_formatters())
        parent_match = (
            self._parent._find_subclass_value_formatter_match(value_type)
            if self._parent is not None
            else None
        )
        if local_match is None:
            return parent_match
        if parent_match is None:
            return local_match
        return local_match if local_match[0] <= parent_match[0] else parent_match

    def _find_subclass_step_formatter_match(
        self,
        operator_type: type[Any],
    ) -> tuple[int, StepFormatter] | None:
        local_match = _best_subclass_formatter_match(operator_type, self.step_formatters())
        parent_match = (
            self._parent._find_subclass_step_formatter_match(operator_type)
            if self._parent is not None
            else None
        )
        if local_match is None:
            return parent_match
        if parent_match is None:
            return local_match
        return local_match if local_match[0] <= parent_match[0] else parent_match
