from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from .region import RegionOpener

_MISSING = object()
_DISPLAY_SAFE = (bool, int, float, str, bytes, type(None))


@dataclass(frozen=True)
class _StaticBoundary:
    input_type: Any
    output_type: Any


@dataclass(frozen=True)
class _DescriptionLayout:
    label_width: int
    input_width: int
    output_width: int
    config_width: int
    show_config: bool
    base_indent: int


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


def _format_step_line(
    step: "StepDescription",
    indent: int = 0,
    layout: _DescriptionLayout | None = None,
) -> str:
    label = (" " * indent) + step.label
    input_type = _format_annotation(step.input_type)
    output_type = _format_annotation(step.output_type)
    config = _format_config(step.operator_config)

    if layout is None:
        line = f"{label}  Input: {input_type}  Output: {output_type}"
        if config:
            line += f"  Cfg: {config}"
        return line

    line = (
        f"{label:<{layout.label_width}} | "
        f"{input_type:<{layout.input_width}} | "
        f"{output_type:<{layout.output_width}}"
    )
    if layout.show_config:
        line += f" | {config:<{layout.config_width}}"
    return line.rstrip()


@dataclass
class StepDescription:
    label: str
    kind: str
    input_type: Any
    output_type: Any
    operator_config: dict[str, Any] = field(default_factory=dict)
    children: list["StepDescription"] = field(default_factory=list)

    def __repr__(self) -> str:
        return _format_step_line(self)


@dataclass
class PipelineDescription:
    steps: list[StepDescription] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = ["PipelineDescription"]
        if self.steps:
            layout = _build_description_layout(self.steps, base_indent=2)
            lines.append("")
            lines.append(_format_table_header(layout))
            lines.append(_format_table_separator(layout))
            self._repr_steps(self.steps, lines, indent=2, layout=layout)
        return "\n".join(lines)

    @staticmethod
    def _repr_steps(
        steps: list[StepDescription],
        lines: list[str],
        indent: int,
        layout: _DescriptionLayout,
    ) -> None:
        for step in steps:
            lines.append(_format_step_line(step, indent=indent, layout=layout))
            if step.children:
                PipelineDescription._repr_steps(step.children, lines, indent + 2, layout)


def _build_pipeline_description(
    *,
    operators: list[Any],
    expand_embedded: bool,
    label_for: Callable[[int], str],
    is_pipeline_operator: Callable[[Any], bool],
) -> PipelineDescription:
    if not operators:
        return PipelineDescription(steps=[])

    boundaries = [_describe_static_boundary(operator) for operator in operators]
    steps = _build_description_steps(
        operators=operators,
        boundaries=boundaries,
        expand_embedded=expand_embedded,
        label_for=label_for,
        is_pipeline_operator=is_pipeline_operator,
    )

    return PipelineDescription(steps=steps)


def _build_description_steps(
    *,
    operators: list[Any],
    boundaries: list[_StaticBoundary],
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
        boundary = boundaries[i]
        label = label_for(i)

        if isinstance(operator, RegionOpener):
            region_end = _find_region_end(
                operators=operators,
                start=i + 1,
                opening_op=type(operator),
                closing_op=operator.closing_type,
                end=end,
            )
            if region_end is not None:
                steps.append(
                    StepDescription(
                        label=label,
                        kind="region",
                        input_type=Any,
                        output_type=Any,
                        operator_config=_describe_operator_config(operator),
                        children=_build_description_steps(
                            operators=operators,
                            boundaries=boundaries,
                            expand_embedded=expand_embedded,
                            label_for=label_for,
                            is_pipeline_operator=is_pipeline_operator,
                            start=i + 1,
                            end=region_end,
                        ),
                    )
                )
                i = region_end + 1
                continue

        if is_pipeline_operator(operator):
            described_pipeline = operator.pipeline._describe(
                expand_embedded=expand_embedded,
            )
            pipeline_boundary = _resolve_embedded_pipeline_boundary(
                operator=operator,
                described_pipeline=described_pipeline,
            )
            steps.append(
                StepDescription(
                    label=label,
                    kind="pipeline",
                    input_type=pipeline_boundary.input_type,
                    output_type=pipeline_boundary.output_type,
                    operator_config=_describe_operator_config(operator),
                    children=described_pipeline.steps if expand_embedded else [],
                )
            )
            i += 1
            continue

        steps.append(
            StepDescription(
                label=label,
                kind="operator",
                input_type=boundary.input_type,
                output_type=boundary.output_type,
                operator_config=_describe_operator_config(operator),
            )
        )
        i += 1

    return steps


def _find_region_end(
    *,
    operators: list[Any],
    start: int,
    opening_op: type,
    closing_op: type,
    end: int,
) -> int | None:
    depth = 1
    j = start
    while j < end:
        operator = operators[j]
        if isinstance(operator, opening_op):
            depth += 1
        elif isinstance(operator, closing_op):
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return None


def _describe_static_boundary(operator: Any) -> _StaticBoundary:
    target = _get_signature_target(operator)
    if target is None:
        return _StaticBoundary(input_type=Any, output_type=Any)

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return _StaticBoundary(input_type=Any, output_type=Any)

    hints = _resolve_type_hints(target)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not parameters:
        input_type = Any
    else:
        input_type = _collapse_input_types(
            tuple(_coerce_annotation(hints.get(parameter.name, Any)) for parameter in parameters)
        )

    return _StaticBoundary(
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


def _description_boundary(description: PipelineDescription) -> _StaticBoundary:
    if not description.steps:
        return _StaticBoundary(input_type=Any, output_type=Any)
    return _StaticBoundary(
        input_type=description.steps[0].input_type,
        output_type=description.steps[-1].output_type,
    )


def _resolve_embedded_pipeline_boundary(
    *,
    operator: Any,
    described_pipeline: PipelineDescription,
) -> _StaticBoundary:
    boundary = _description_boundary(described_pipeline)
    if boundary.input_type is not Any and boundary.output_type is not Any:
        return boundary

    contract = _infer_pipeline_contract(getattr(operator, "pipeline", None))
    if contract is None:
        return boundary

    return _StaticBoundary(
        input_type=contract.input_type if boundary.input_type is Any else boundary.input_type,
        output_type=contract.output_type if boundary.output_type is Any else boundary.output_type,
    )


def _infer_pipeline_contract(pipeline: Any) -> Any | None:
    if pipeline is None:
        return None

    validate = getattr(pipeline, "validate", None)
    if not callable(validate):
        return None

    try:
        return validate(inference=True)
    except Exception:
        return None


def _build_description_layout(
    steps: list[StepDescription],
    *,
    base_indent: int,
) -> _DescriptionLayout:
    label_width = len((" " * base_indent) + "Step")
    input_width = len("Input")
    output_width = len("Output")
    config_width = len("Cfg")
    show_config = False

    def visit(items: list[StepDescription], indent: int) -> None:
        nonlocal label_width, input_width, output_width, config_width, show_config
        for step in items:
            label_width = max(label_width, len((" " * indent) + step.label))
            input_width = max(input_width, len(_format_annotation(step.input_type)))
            output_width = max(output_width, len(_format_annotation(step.output_type)))
            config_width = max(config_width, len(_format_config(step.operator_config)))
            show_config = show_config or bool(step.operator_config)
            if step.children:
                visit(step.children, indent + 2)

    visit(steps, base_indent)
    return _DescriptionLayout(
        label_width=label_width,
        input_width=input_width,
        output_width=output_width,
        config_width=config_width,
        show_config=show_config,
        base_indent=base_indent,
    )


def _format_table_header(layout: _DescriptionLayout) -> str:
    line = (
        f"{(' ' * layout.base_indent) + 'Step':<{layout.label_width}} | "
        f"{'Input':<{layout.input_width}} | "
        f"{'Output':<{layout.output_width}}"
    )
    if layout.show_config:
        line += f" | {'Cfg':<{layout.config_width}}"
    return line


def _format_table_separator(layout: _DescriptionLayout) -> str:
    line = (
        f"{(' ' * layout.base_indent) + ('-' * (layout.label_width - layout.base_indent))}"
        f"-+-{'-' * layout.input_width}"
        f"-+-{'-' * layout.output_width}"
    )
    if layout.show_config:
        line += f"-+-{'-' * layout.config_width}"
    return line


def _format_config(config: dict[str, Any]) -> str:
    return repr(config) if config else ""


def _describe_operator_config(operator: Any) -> dict[str, Any]:
    if inspect.isfunction(operator) or inspect.ismethod(operator):
        return {}

    try:
        signature = inspect.signature(type(operator).__init__)
    except (TypeError, ValueError):
        return {}

    config: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.name == "pipeline":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue

        value = _lookup_constructor_arg(operator, parameter.name)
        if value is _MISSING or value is None:
            continue
        config[parameter.name] = _normalize_config_value(value)

    return config


def _lookup_constructor_arg(operator: Any, name: str) -> Any:
    for owner in (operator, getattr(operator, "gate", None), getattr(operator, "_inner", None)):
        if owner is None:
            continue
        for attr_name in (name, f"_{name}"):
            if hasattr(owner, attr_name):
                return getattr(owner, attr_name)
    return _lookup_derived_constructor_arg(operator, name)


def _lookup_derived_constructor_arg(operator: Any, name: str) -> Any:
    if hasattr(operator, "_mapping"):
        mapping = getattr(operator, "_mapping")
        if isinstance(mapping, dict):
            names = tuple(mapping)
            aliases = tuple(mapping.values())
            if name == "names":
                return _collapse_variadic_value(names)
            if name == "as_" and aliases != names:
                return _collapse_variadic_value(aliases)

    if name == "providers":
        session = getattr(operator, "session", None)
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            try:
                return tuple(get_providers())
            except Exception:
                return _MISSING

    if name == "dtype" and hasattr(operator, "model_dtype"):
        dtype = getattr(operator, "model_dtype")
        if dtype is None:
            return None
        dtype_name = getattr(dtype, "name", None)
        if isinstance(dtype_name, str):
            return dtype_name
        return str(dtype)

    if name == "serialize" and hasattr(operator, "_lock"):
        return hasattr(getattr(operator, "_lock"), "acquire")

    return _MISSING


def _collapse_variadic_value(values: tuple[Any, ...]) -> Any:
    if len(values) == 1:
        return values[0]
    return values


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, _DISPLAY_SAFE):
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_config_value(item) for item in value)
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_config_value(item) for key, item in value.items()}
    return repr(value)


def _strip_module_qualifiers(text: str) -> str:
    return re.sub(r"\b(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)\b", r"\1", text)
