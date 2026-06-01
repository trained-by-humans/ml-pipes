from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from .operator import get_operator_args
from .region import RegionOpener


# ---------------------------------------------------------------------------
# Description model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StepBoundary:
    input_type: Any
    output_type: Any


@dataclass(frozen=True)
class _TableLayout:
    label_width: int
    input_width: int
    output_width: int
    args_width: int
    show_args: bool
    base_indent: int


@dataclass(init=False)
class StepDescription:
    label: str
    kind: str
    input_type: Any
    output_type: Any
    operator_args: dict[str, Any] = field(default_factory=dict)
    children: list["StepDescription"] = field(default_factory=list)

    def __init__(
        self,
        label: str,
        kind: str,
        input_type: Any,
        output_type: Any,
        operator_args: dict[str, Any] | None = None,
        *,
        operator_config: dict[str, Any] | None = None,
        children: list["StepDescription"] | None = None,
    ) -> None:
        self.label = label
        self.kind = kind
        self.input_type = input_type
        self.output_type = output_type
        self.operator_args = operator_args if operator_args is not None else (
            operator_config if operator_config is not None else {}
        )
        self.children = children if children is not None else []

    def __repr__(self) -> str:
        return _format_step_line(self)

    @property
    def operator_config(self) -> dict[str, Any]:
        return self.operator_args


@dataclass
class PipelineDescription:
    steps: list[StepDescription] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["PipelineDescription"]
        if self.steps:
            layout = _build_table_layout(self.steps, base_indent=2)
            lines.append("")
            lines.append(_format_table_header(layout))
            lines.append(_format_table_separator(layout))
            self._append_step_lines(self.steps, lines, indent=2, layout=layout)
        return "\n".join(lines)

    @staticmethod
    def _append_step_lines(
        steps: list[StepDescription],
        lines: list[str],
        indent: int,
        layout: _TableLayout,
    ) -> None:
        for step in steps:
            lines.append(_format_step_line(step, indent=indent, layout=layout))
            if step.children:
                PipelineDescription._append_step_lines(step.children, lines, indent + 2, layout)


# ---------------------------------------------------------------------------
# Description tree construction
# ---------------------------------------------------------------------------


def _build_pipeline_description(
    *,
    operators: list[Any],
    expand_embedded: bool,
    label_for: Callable[[int], str],
    is_pipeline_operator: Callable[[Any], bool],
) -> PipelineDescription:
    if not operators:
        return PipelineDescription(steps=[])

    step_boundaries = [_describe_operator_boundary(operator) for operator in operators]
    steps = _build_description_steps(
        operators=operators,
        step_boundaries=step_boundaries,
        expand_embedded=expand_embedded,
        label_for=label_for,
        is_pipeline_operator=is_pipeline_operator,
    )
    return PipelineDescription(steps=steps)


def _build_description_steps(
    *,
    operators: list[Any],
    step_boundaries: list[_StepBoundary],
    expand_embedded: bool,
    label_for: Callable[[int], str],
    is_pipeline_operator: Callable[[Any], bool],
    start: int = 0,
    end: int | None = None,
) -> list[StepDescription]:
    end = len(operators) if end is None else end
    steps: list[StepDescription] = []
    i = start

    while i < end:
        operator = operators[i]
        label = label_for(i)
        operator_args = _describe_operator_args(operator)
        step_boundary = step_boundaries[i]

        if isinstance(operator, RegionOpener):
            matching_end = _find_matching_region_end(
                operators=operators,
                start=i + 1,
                opening_op=type(operator),
                closing_op=operator.closing_type,
                end=end,
            )
            if matching_end is not None:
                steps.append(
                    StepDescription(
                        label=label,
                        kind="region",
                        input_type=Any,
                        output_type=Any,
                        operator_args=operator_args,
                        children=_build_description_steps(
                            operators=operators,
                            step_boundaries=step_boundaries,
                            expand_embedded=expand_embedded,
                            label_for=label_for,
                            is_pipeline_operator=is_pipeline_operator,
                            start=i + 1,
                            end=matching_end,
                        ),
                    )
                )
                i = matching_end + 1
                continue

        if is_pipeline_operator(operator):
            embedded_description = operator.pipeline._describe(
                expand_embedded=expand_embedded,
            )
            embedded_boundary = _resolve_embedded_pipeline_boundary(
                operator=operator,
                embedded_description=embedded_description,
            )
            steps.append(
                StepDescription(
                    label=label,
                    kind="pipeline",
                    input_type=embedded_boundary.input_type,
                    output_type=embedded_boundary.output_type,
                    operator_args=operator_args,
                    children=embedded_description.steps if expand_embedded else [],
                )
            )
            i += 1
            continue

        steps.append(
            StepDescription(
                label=label,
                kind="operator",
                input_type=step_boundary.input_type,
                output_type=step_boundary.output_type,
                operator_args=operator_args,
            )
        )
        i += 1

    return steps


def _find_matching_region_end(
    *,
    operators: list[Any],
    start: int,
    opening_op: type,
    closing_op: type,
    end: int,
) -> int | None:
    depth = 1
    i = start
    while i < end:
        operator = operators[i]
        if isinstance(operator, opening_op):
            depth += 1
        elif isinstance(operator, closing_op):
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _resolve_embedded_pipeline_boundary(
    *,
    operator: Any,
    embedded_description: PipelineDescription,
) -> _StepBoundary:
    boundary = _pipeline_boundary_from_description(embedded_description)
    if boundary.input_type is not Any and boundary.output_type is not Any:
        return boundary

    contract = _try_infer_pipeline_contract(getattr(operator, "pipeline", None))
    if contract is None:
        return boundary

    return _StepBoundary(
        input_type=contract.input_type if boundary.input_type is Any else boundary.input_type,
        output_type=contract.output_type if boundary.output_type is Any else boundary.output_type,
    )


def _pipeline_boundary_from_description(description: PipelineDescription) -> _StepBoundary:
    if not description.steps:
        return _StepBoundary(input_type=Any, output_type=Any)
    return _StepBoundary(
        input_type=description.steps[0].input_type,
        output_type=description.steps[-1].output_type,
    )


def _try_infer_pipeline_contract(pipeline: Any) -> Any | None:
    if pipeline is None:
        return None

    validate = getattr(pipeline, "validate", None)
    if not callable(validate):
        return None

    try:
        return validate(inference=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Static boundary inference
# ---------------------------------------------------------------------------


def _describe_operator_boundary(operator: Any) -> _StepBoundary:
    target = _get_signature_target(operator)
    if target is None:
        return _StepBoundary(input_type=Any, output_type=Any)

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return _StepBoundary(input_type=Any, output_type=Any)

    hints = _resolve_type_hints(target)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    input_type = _collapse_input_types(
        tuple(_coerce_annotation(hints.get(parameter.name, Any)) for parameter in parameters)
    )
    return _StepBoundary(
        input_type=input_type,
        output_type=_coerce_annotation(hints.get("return", Any)),
    )


def _get_signature_target(operator: Any) -> Any | None:
    if inspect.isfunction(operator) or inspect.ismethod(operator):
        return operator
    return getattr(operator, "__call__", None)


def _resolve_type_hints(target: Any) -> dict[str, Any]:
    try:
        return get_type_hints(target)
    except Exception:
        raw_hints = getattr(target, "__annotations__", {}) or {}
        return {name: _coerce_annotation(annotation) for name, annotation in raw_hints.items()}


def _coerce_annotation(annotation: Any) -> Any:
    if isinstance(annotation, str):
        return Any
    return annotation


def _collapse_input_types(input_types: tuple[Any, ...]) -> Any:
    if not input_types:
        return Any
    if len(input_types) == 1:
        return input_types[0]
    return tuple[input_types]


# ---------------------------------------------------------------------------
# Operator argument extraction
# ---------------------------------------------------------------------------


def _describe_operator_args(operator: Any) -> dict[str, Any]:
    if inspect.isfunction(operator) or inspect.ismethod(operator):
        return {}
    return get_operator_args(operator)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _build_table_layout(
    steps: list[StepDescription],
    *,
    base_indent: int,
) -> _TableLayout:
    label_width = len((" " * base_indent) + "Step")
    input_width = len("Input")
    output_width = len("Output")
    args_width = len("Args")
    show_args = False

    def visit(items: list[StepDescription], indent: int) -> None:
        nonlocal label_width, input_width, output_width, args_width, show_args
        for step in items:
            label_width = max(label_width, len((" " * indent) + step.label))
            input_width = max(input_width, len(_format_annotation(step.input_type)))
            output_width = max(output_width, len(_format_annotation(step.output_type)))
            args_width = max(args_width, len(_format_args(step.operator_args)))
            show_args = show_args or bool(step.operator_args)
            if step.children:
                visit(step.children, indent + 2)

    visit(steps, base_indent)
    return _TableLayout(
        label_width=label_width,
        input_width=input_width,
        output_width=output_width,
        args_width=args_width,
        show_args=show_args,
        base_indent=base_indent,
    )


def _format_step_line(
    step: StepDescription,
    indent: int = 0,
    layout: _TableLayout | None = None,
) -> str:
    label = (" " * indent) + step.label
    input_type = _format_annotation(step.input_type)
    output_type = _format_annotation(step.output_type)
    args = _format_args(step.operator_args)

    if layout is None:
        line = f"{label}  Input: {input_type}  Output: {output_type}"
        if args:
            line += f"  Args: {args}"
        return line

    line = (
        f"{label:<{layout.label_width}} | "
        f"{input_type:<{layout.input_width}} | "
        f"{output_type:<{layout.output_width}}"
    )
    if layout.show_args:
        line += f" | {args:<{layout.args_width}}"
    return line.rstrip()


def _format_table_header(layout: _TableLayout) -> str:
    line = (
        f"{(' ' * layout.base_indent) + 'Step':<{layout.label_width}} | "
        f"{'Input':<{layout.input_width}} | "
        f"{'Output':<{layout.output_width}}"
    )
    if layout.show_args:
        line += f" | {'Args':<{layout.args_width}}"
    return line


def _format_table_separator(layout: _TableLayout) -> str:
    line = (
        f"{(' ' * layout.base_indent) + ('-' * (layout.label_width - layout.base_indent))}"
        f"-+-{'-' * layout.input_width}"
        f"-+-{'-' * layout.output_width}"
    )
    if layout.show_args:
        line += f"-+-{'-' * layout.args_width}"
    return line


def _format_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    return "{" + ", ".join(f"{key!r}: {_format_arg_value(value)}" for key, value in args.items()) + "}"


def _format_arg_value(value: Any) -> str:
    if isinstance(value, (bool, int, float, str, bytes, type(None))):
        return repr(value)
    if isinstance(value, tuple):
        items = ", ".join(_format_arg_value(item) for item in value)
        if len(value) == 1:
            items += ","
        return f"({items})"
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
        items = sorted((_format_arg_value(item) for item in value), key=str)
        return "{" + ", ".join(items) + "}"
    if isinstance(value, frozenset):
        if not value:
            return "frozenset()"
        items = sorted((_format_arg_value(item) for item in value), key=str)
        return "frozenset({" + ", ".join(items) + "})"
    if callable(value):
        return _callable_label(value)
    return repr(value)


def _callable_label(value: Any) -> str:
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return type(value).__qualname__


def _format_annotation(annotation: Any) -> str:
    if annotation is Any:
        return "Any"
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    if isinstance(annotation, tuple):
        return "(" + ", ".join(_format_annotation(arg) for arg in annotation) + ")"

    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"({_format_annotation(args[0])}, ...)"
        return "(" + ", ".join(_format_annotation(arg) for arg in args) + ")"
    if origin is list:
        args = get_args(annotation)
        if not args:
            return "list"
        return f"list[{', '.join(_format_annotation(arg) for arg in args)}]"
    if origin is dict:
        args = get_args(annotation)
        if not args:
            return "dict"
        return f"dict[{', '.join(_format_annotation(arg) for arg in args)}]"
    if origin is set:
        args = get_args(annotation)
        if not args:
            return "set"
        return f"set[{', '.join(_format_annotation(arg) for arg in args)}]"
    if origin in (UnionType, Union):
        return " | ".join(_format_annotation(arg) for arg in get_args(annotation))

    return _strip_module_qualifiers(str(annotation).replace("typing.", ""))


def _strip_module_qualifiers(text: str) -> str:
    return re.sub(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)\b", r"\1", text)
