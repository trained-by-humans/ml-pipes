from __future__ import annotations

from typing import Any

from ml_pipes.inspection.registry import FormatterRegistry, StepFormatter, ValueFormatter, ValueT

_GLOBAL_FORMATTER_REGISTRY = FormatterRegistry()


def global_formatter_registry() -> FormatterRegistry:
    return _GLOBAL_FORMATTER_REGISTRY


def register_value_formatter(
    value_type: type[ValueT],
    formatter: ValueFormatter[ValueT],
    *,
    allow_override: bool = False,
) -> None:
    _GLOBAL_FORMATTER_REGISTRY.register_value_formatter(
        value_type,
        formatter,
        allow_override=allow_override,
    )


def register_step_formatter(
    operator_type: type[Any],
    formatter: StepFormatter,
    *,
    allow_override: bool = False,
) -> None:
    _GLOBAL_FORMATTER_REGISTRY.register_step_formatter(
        operator_type,
        formatter,
        allow_override=allow_override,
    )
