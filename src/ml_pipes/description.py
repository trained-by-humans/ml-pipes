from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .operator import get_operator_args, get_operator_constructor_signature


@dataclass(frozen=True)
class StepDescription:
    label: str
    operator_args: dict[str, Any] = field(default_factory=dict)
    children: list["StepDescription"] = field(default_factory=list)
    kind: str = "operator"
    _constructor_signature: inspect.Signature | None = field(default=None, repr=False, compare=False)

    def render(self, indent: int = 0) -> str:
        lines = [_format_step_line(self, indent=indent)]
        for child_index, child in enumerate(self.children):
            lines.extend(_render_step_lines(child, indent + 2, index=child_index))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__


@dataclass(frozen=True)
class PipelineDescription:
    steps: list[StepDescription] = field(default_factory=list)

    def render(self) -> str:
        if not self.steps:
            return "Pipeline[]"

        lines = ["Pipeline["]
        for index, step in enumerate(self.steps):
            lines.extend(_render_step_lines(step, indent=2, index=index))
        lines.append("]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.render()

    __str__ = __repr__


def _build_pipeline_description(
    operators: Iterable[Any],
    expand_embedded: bool = True,
    is_embedded_operator: Callable[[Any], bool] = lambda _operator: False,
) -> PipelineDescription:
    return PipelineDescription(
        steps=_build_description_steps(
            operators=operators,
            expand_embedded=expand_embedded,
            is_embedded_operator=is_embedded_operator,
        )
    )


def _build_description_steps(
    operators: Iterable[Any],
    expand_embedded: bool,
    is_embedded_operator: Callable[[Any], bool],
) -> list[StepDescription]:
    steps: list[StepDescription] = []
    for operator in operators:
        is_embedded = is_embedded_operator(operator)
        children: list[StepDescription] = []
        if is_embedded and expand_embedded:
            pipeline = getattr(operator, "pipeline", None)
            inner_operators = getattr(pipeline, "operators", [])
            children = _build_description_steps(
                operators=inner_operators,
                expand_embedded=expand_embedded,
                is_embedded_operator=is_embedded_operator,
            )

        steps.append(
            StepDescription(
                label=_describe_operator_name(operator),
                operator_args=get_operator_args(operator),
                children=children,
                kind="pipeline" if is_embedded else "operator",
                _constructor_signature=_describe_operator_constructor_signature(operator),
            )
        )
    return steps


def _describe_operator_name(operator: Any) -> str:
    if inspect.isfunction(operator) or inspect.ismethod(operator) or inspect.isbuiltin(operator):
        return getattr(operator, "__name__", type(operator).__name__)
    return type(operator).__name__


def _describe_operator_constructor_signature(operator: Any) -> inspect.Signature | None:
    return get_operator_constructor_signature(operator)


def _render_step_lines(
    step: StepDescription,
    indent: int = 0,
    index: int | None = None,
) -> list[str]:
    lines = [_format_step_line(step, indent=indent, index=index)]
    for child_index, child in enumerate(step.children):
        lines.extend(_render_step_lines(child, indent + 2, index=child_index))
    return lines


def _format_step_line(
    step: StepDescription,
    indent: int = 0,
    index: int | None = None,
) -> str:
    prefix = f"{index}:" if index is not None else ""
    args = _format_call_arguments(step.operator_args, step._constructor_signature)
    return f"{' ' * indent}{prefix}{step.label}({args})"


def _format_call_arguments(
    operator_args: dict[str, Any],
    constructor_signature: inspect.Signature | None,
) -> str:
    parts: list[str] = []
    if not operator_args:
        return ""

    if constructor_signature is None:
        parts.extend(
            f"{name}={_format_arg_value(value)}"
            for name, value in operator_args.items()
        )
        return ", ".join(parts)

    consumed: set[str] = set()
    for parameter in constructor_signature.parameters.values():
        if parameter.name not in operator_args:
            continue
        consumed.add(parameter.name)
        value = operator_args[parameter.name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            parts.append(_format_arg_value(value))
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.extend(_format_arg_value(item) for item in value)
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parts.extend(f"{name}={_format_arg_value(item)}" for name, item in value.items())
            continue
        parts.append(f"{parameter.name}={_format_arg_value(value)}")

    for name, value in operator_args.items():
        if name in consumed:
            continue
        parts.append(f"{name}={_format_arg_value(value)}")
    return ", ".join(parts)


def _format_arg_value(value: Any) -> str:
    if isinstance(value, tuple):
        if len(value) == 1:
            return f"({_format_arg_value(value[0])},)"
        return "(" + ", ".join(_format_arg_value(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_format_arg_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_format_arg_value(key)}: {_format_arg_value(item)}"
            for key, item in value.items()
        ) + "}"
    if isinstance(value, set):
        if not value:
            return "set()"
        items = sorted(_format_arg_value(item) for item in value)
        return "{" + ", ".join(items) + "}"
    if isinstance(value, frozenset):
        if not value:
            return "frozenset()"
        items = sorted(_format_arg_value(item) for item in value)
        return "frozenset({" + ", ".join(items) + "})"
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return repr(value)
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return _callable_label(value)
    if inspect.isclass(value):
        return value.__name__
    if callable(value):
        return _callable_label(value)
    return repr(value)


def _callable_label(value: Any) -> str:
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return getattr(value, "__name__", type(value).__name__)
    if inspect.isclass(value):
        return value.__name__
    return type(value).__name__
