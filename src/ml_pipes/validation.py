from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, get_args, get_origin, get_type_hints

from .context import ContextOp, Recall, Store
from .region import RegionCloser, RegionOpener

_UNBOUND = object()


class PipelineValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TypeContract:
    input_type: Any
    output_type: Any


@dataclass(frozen=True)
class _BoundarySignature:
    input_types: tuple[Any, ...]
    output_type: Any


@dataclass
class _OperatorBoundary:
    operator: Any
    previous_output_type: Any
    context_inputs: dict[str, Any] | None
    dynamic_boundary: _BoundarySignature | None
    static_boundary: _BoundarySignature | None

    @property
    def effective_boundary(self) -> _BoundarySignature:
        return self.dynamic_boundary or self.static_boundary

    @property
    def input_types(self) -> tuple[Any, ...]:
        return self.effective_boundary.input_types

    @property
    def output_type(self) -> Any:
        return self.effective_boundary.output_type

class PipelineValidator:
    def __init__(self, operators: list[Any]):
        self.operators = operators

    def validate(self, strict: bool = False) -> TypeContract:
        self._validate_regions()
        self._validate_context_interactions()
        return self._resolve_type_contract(strict=strict)

    @staticmethod
    def _label_for(i: int, operator: Any) -> str:
        name = operator.__name__ if inspect.isfunction(operator) or inspect.ismethod(operator) else type(operator).__name__
        return f"{i}:{name}"

    def _validate_regions(self) -> None:
        stack: list[tuple[RegionOpener, int]] = []
        for i, op in enumerate(self.operators):
            match op:
                case RegionOpener() if stack and type(stack[-1][0]) is type(op):
                    raise PipelineValidationError(
                        f"Directly nested {type(op).__name__} regions are not supported — "
                        f"a {type(op).__name__} region may not open inside another {type(op).__name__} region"
                    )
                case RegionOpener():
                    stack.append((op, i))
                case RegionCloser() if not stack:
                    raise PipelineValidationError(
                        f"{type(op).__name__} at position {i} has no matching opener"
                    )
                case RegionCloser() if not isinstance(op, stack[-1][0].closing_type):
                    top_opener, top_pos = stack[-1]
                    raise PipelineValidationError(
                        f"{type(op).__name__} at position {i} closes {type(top_opener).__name__} "
                        f"opened at position {top_pos} — regions cannot interleave"
                    )
                case RegionCloser():
                    stack.pop()

        for opener, pos in stack:
            raise PipelineValidationError(
                f"{type(opener).__name__} at position {pos} has no matching {opener.closing_type.__name__}"
            )

    def _validate_context_interactions(self) -> None:
        stored_keys: set[str] = set()
        stack: list[set[str]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, RegionOpener):
                stack.append(stored_keys)
                stored_keys = set()
            elif isinstance(operator, RegionCloser):
                stored_keys = stack.pop()
            elif isinstance(operator, Store):
                stored_keys.add(operator.name)
            elif isinstance(operator, Recall):
                if operator.name not in stored_keys:
                    available = sorted(stored_keys)
                    raise PipelineValidationError(
                        f"Recall({operator.name!r}) at {self._label_for(i, operator)} "
                        f"references a key that was not stored. "
                        f"Keys available at this point: {available if available else '(none)'}"
                    )

    def _resolve_type_contract(self, strict: bool = False) -> TypeContract:
        boundaries = self._resolve_operator_boundaries()
        self._validate_downstream_compatibility(boundaries)
        resolved_input_type = self._refine_operator_boundaries(boundaries)

        if strict:
            self._validate_contracts_strictly(boundaries)

        return TypeContract(input_type=resolved_input_type, output_type=boundaries[-1].output_type)

    def _resolve_operator_boundaries(self) -> list[_OperatorBoundary]:
        boundaries: list[_OperatorBoundary] = []
        previous_output_type: Any = Any
        stored_annotations: dict[str, Any] = {}
        stack: list[dict[str, Any]] = []

        for i, operator in enumerate(self.operators):
            if isinstance(operator, RegionOpener):
                stack.append(stored_annotations)
                stored_annotations = {}
            elif isinstance(operator, RegionCloser):
                stored_annotations = stack.pop()

            input_context = dict(stored_annotations) if isinstance(operator, ContextOp) else None
            dynamic_boundary = self._resolve_dynamic_boundary(operator, previous_output_type, stored_annotations)
            static_boundary = self._resolve_static_boundary(operator)
            if dynamic_boundary is None and static_boundary is None:
                resolve_operator_contract(operator)

            boundaries.append(
                _OperatorBoundary(
                    operator=operator,
                    previous_output_type=previous_output_type,
                    context_inputs=input_context,
                    dynamic_boundary=dynamic_boundary,
                    static_boundary=static_boundary,
                )
            )
            previous_output_type = boundaries[-1].output_type

        return boundaries

    def _validate_downstream_compatibility(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if is_annotation_compatible(boundary.previous_output_type, boundary.input_types):
                continue

            previous_name = type(boundaries[i - 1].operator).__name__
            operator = boundary.operator
            raise PipelineValidationError(
                f"Pipeline contract mismatch at {self._label_for(i, operator)}: "
                f"{previous_name} returns {format_annotation(boundary.previous_output_type)} "
                f"but {operator.__class__.__name__} expects {format_parameter_annotations(boundary.input_types)}"
            )

    @classmethod
    def _refine_operator_boundaries(cls, boundaries: list[_OperatorBoundary]) -> Any:
        downstream_required_input: Any = Any
        for boundary in reversed(boundaries):
            # Example: static says `object`, dynamic says `tuple[int, Any]` -> start from `tuple[int, Any]`.
            local_input_type = cls._collapse_boundary_input_types(boundary)
            # Example: downstream needs `tuple[int, str]` -> project that shape backward through this operator.
            projected_input = cls._project_input_back_through(boundary, downstream_required_input)
            # Example: local says `tuple[int, Any]`, projection says `tuple[int, str]` -> keep `tuple[int, str]`.
            downstream_required_input = cls._refine_input_constraint(local_input_type, projected_input)
        return downstream_required_input

    @staticmethod
    def _collapse_input_types(input_types: tuple[Any, ...]) -> Any:
        if len(input_types) == 1:
            return input_types[0]
        return tuple[input_types]

    @classmethod
    def _collapse_boundary_input_types(cls, boundary: _OperatorBoundary) -> Any:
        static_input = (
            cls._collapse_input_types(boundary.static_boundary.input_types)
            if boundary.static_boundary is not None
            else None
        )
        dynamic_input = (
            cls._collapse_input_types(boundary.dynamic_boundary.input_types)
            if boundary.dynamic_boundary is not None
            else None
        )

        match (static_input, dynamic_input):
            case (None, input_type):
                return input_type
            case (input_type, None):
                return input_type
            case (static_input, dynamic_input):
                return cls._refine_input_constraint(static_input, dynamic_input)

    @classmethod
    def _project_input_back_through(cls, boundary: _OperatorBoundary, inferred: Any) -> Any:
        contract_input = cls._project_contract_input_back_through(boundary, inferred)
        collapsed_input = cls._collapse_boundary_input_types(boundary)
        if contract_input is not Any:
            return cls._refine_input_constraint(collapsed_input, contract_input)
        if is_annotation_compatible(boundary.output_type, (inferred,)):
            return collapsed_input
        return Any

    @classmethod
    def _project_contract_input_back_through(cls, boundary: _OperatorBoundary, inferred: Any) -> Any:
        if boundary.dynamic_boundary is None:
            return Any

        template_input = cls._collapse_input_types(boundary.dynamic_boundary.input_types)
        specialized_input = cls._specialize_input_from_output_template(
            template_input,
            boundary.dynamic_boundary.output_type,
            inferred,
        )
        if specialized_input is not Any and cls._confirm_contract_projection(boundary, specialized_input, inferred):
            return specialized_input

        contract_probe = cls._probe_contract(boundary, inferred)
        if contract_probe is None:
            return Any
        input_types, output_type = contract_probe
        collapsed_input = cls._collapse_input_types(input_types)
        if output_type == inferred and (collapsed_input is Any or collapsed_input == inferred):
            return inferred
        return Any

    @classmethod
    def _confirm_contract_projection(cls, boundary: _OperatorBoundary, candidate_input: Any, inferred: Any) -> bool:
        contract_probe = cls._probe_contract(boundary, candidate_input)
        if contract_probe is None:
            return False
        _, output_type = contract_probe
        return output_type == inferred

    @classmethod
    def _probe_contract(
        cls,
        boundary: _OperatorBoundary,
        probe_input: Any,
    ) -> tuple[tuple[Any, ...], Any] | None:
        probe_annotations = dict(boundary.context_inputs or {})
        try:
            return boundary.operator.resolve_contract(
                probe_input,
                probe_annotations,
                expand_output_annotation,
                None,
            )
        except Exception:
            return None

    @classmethod
    def _specialize_input_from_output_template(cls, input_template: Any, output_template: Any, inferred_output: Any) -> Any:
        binding = cls._bind_any_placeholder(output_template, inferred_output, None)
        if binding is _UNBOUND:
            return Any
        return cls._replace_any_placeholder(input_template, binding)

    @classmethod
    def _bind_any_placeholder(cls, template: Any, value: Any, binding: Any) -> Any:
        # Example: template=`Any`, value=`int`, binding=None -> first placeholder binds to `int`.
        if template is Any and binding is None:
            return value
        # Example: template=`Any`, value=`int`, binding=`int` -> placeholder stays bound to `int`.
        if template is Any and binding == value:
            return binding
        # Example: template=`Any`, value=`str`, binding=`int` -> conflicting binding, so fail.
        if template is Any:
            return _UNBOUND
        # Example: template=`tuple[int, str]`, value=`tuple[int, str]` -> exact structure match.
        if template == value:
            return binding
        template_origin = get_origin(template)
        value_origin = get_origin(value)
        # Example: template=`tuple[Any, str]`, value=`list[int, str]` -> different origins, so fail.
        if template_origin is None or template_origin != value_origin:
            return _UNBOUND
        template_args = get_args(template)
        value_args = get_args(value)
        # Example: template=`tuple[Any, str]`, value=`tuple[int, str, float]` -> different arity, so fail.
        if len(template_args) != len(value_args):
            return _UNBOUND
        return cls._bind_any_placeholder_args(template_args, value_args, binding)

    @classmethod
    def _bind_any_placeholder_args(cls, template_args: tuple[Any, ...], value_args: tuple[Any, ...], binding: Any) -> Any:
        for template_arg, value_arg in zip(template_args, value_args, strict=True):
            binding = cls._bind_any_placeholder(template_arg, value_arg, binding)
            if binding is _UNBOUND:
                return _UNBOUND
        return binding

    @classmethod
    def _replace_any_placeholder(cls, template: Any, binding: Any) -> Any:
        if template is Any:
            return Any if binding is None else binding
        origin = get_origin(template)
        if origin is None:
            return template
        args = tuple(cls._replace_any_placeholder(arg, binding) for arg in get_args(template))
        if len(args) == 1:
            return origin[args[0]]
        return origin[args]

    @classmethod
    def _refine_input_constraint(cls, current: Any, candidate: Any) -> Any:
        if cls._can_refine_annotation(current, candidate):
            return candidate
        return current

    @classmethod
    def _can_refine_annotation(cls, current: Any, candidate: Any) -> bool:
        # Example: current=`tuple[int, Any]`, candidate=`Any` -> vague candidate cannot refine.
        if candidate is Any or candidate is None:
            return False
        # Example: current=`Any`, candidate=`tuple[int, str]` -> any concrete structure refines `Any`.
        if current is Any or current is None:
            return True
        # Example: current=`tuple[int, str]`, candidate=`tuple[int, str]` -> no refinement needed.
        if candidate == current:
            return False
        # Example: current=`object`, candidate=`TensorPayload` -> concrete subtype refines broad `object`.
        if current is object:
            return True
        if isinstance(candidate, type) and isinstance(current, type):
            try:
                return issubclass(candidate, current)
            except TypeError:
                return False
        candidate_origin = get_origin(candidate)
        current_origin = get_origin(current)
        # Example: current=`tuple[int, Any]`, candidate=`list[int, str]` -> different container kinds cannot refine.
        if candidate_origin != current_origin or candidate_origin is None:
            return False
        candidate_args = get_args(candidate)
        current_args = get_args(current)
        # Example: current=`tuple[int, Any]`, candidate=`tuple[int, str, float]` -> different arity cannot refine.
        if len(candidate_args) != len(current_args):
            return False

        changed = False
        for cand_arg, curr_arg in zip(candidate_args, current_args, strict=True):
            if curr_arg is Any:
                if cand_arg is Any:
                    continue
                changed = True
                continue
            if cand_arg is Any:
                return False
            if cand_arg == curr_arg:
                continue
            if cls._can_refine_annotation(curr_arg, cand_arg):
                changed = True
                continue
            return False
        return changed

    @staticmethod
    def _resolve_dynamic_boundary(
        operator: Any,
        previous_output_type: Any,
        stored_annotations: dict[str, Any],
    ) -> _BoundarySignature | None:
        if not hasattr(operator, "resolve_contract"):
            return None
        input_types, output_type = operator.resolve_contract(
            previous_output_type,
            stored_annotations,
            expand_output_annotation,
            PipelineValidationError,
        )
        return _BoundarySignature(input_types=input_types, output_type=output_type)

    @staticmethod
    def _resolve_static_boundary(operator: Callable[..., Any]) -> _BoundarySignature | None:
        try:
            input_types, output_type = resolve_operator_contract(operator)
        except Exception:
            return None
        return _BoundarySignature(input_types=input_types, output_type=output_type)

    def _validate_contracts_strictly(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if boundary.dynamic_boundary is None and any(not is_concrete(t) for t in boundary.input_types):
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: input type is unresolved (Any).\n"
                    f"  Fix: annotate the parameter with a concrete type, or implement resolve_contract "
                    f"to accept and thread the upstream type dynamically."
                )
            if not is_concrete(boundary.output_type) and not self._is_explicitly_transitive_boundary(boundary):
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: output type is unresolved (Any).\n"
                    f"  Fix: annotate the return type with a concrete type, or implement resolve_contract "
                    f"to return the upstream type (e.g. passthrough: return (Any,), current_output)."
                )

    @classmethod
    def _is_explicitly_transitive_boundary(cls, boundary: _OperatorBoundary) -> bool:
        if boundary.dynamic_boundary is None:
            return False
        probe_input = cls._build_transitivity_probe_input(boundary)
        if probe_input is Any:
            return False
        result = cls._probe_contract(boundary, probe_input)
        if result is None:
            return False
        _, probe_output = result
        return cls._can_refine_annotation(boundary.output_type, probe_output)

    @classmethod
    def _build_transitivity_probe_input(cls, boundary: _OperatorBoundary) -> Any:
        probe_from_previous = cls._materialize_probe_annotation(boundary.previous_output_type)
        if probe_from_previous is not Any:
            return probe_from_previous

        return cls._materialize_probe_annotation(
            cls._collapse_input_types(boundary.dynamic_boundary.input_types)
        )

    @classmethod
    def _materialize_probe_annotation(cls, annotation: Any) -> Any:
        if annotation is Any or annotation is None or annotation is object:
            return int

        origin = get_origin(annotation)
        if origin is None:
            return annotation

        args = tuple(cls._materialize_probe_annotation(arg) for arg in get_args(annotation))
        if len(args) == 1:
            return origin[args[0]]
        return origin[args]


def resolve_operator_contract(operator: Callable[..., Any]) -> tuple[tuple[Any, ...], Any]:
    target = get_signature_target(operator)
    hints = get_type_hints(target)
    signature = inspect.signature(target)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if not parameters:
        raise PipelineValidationError(
            f"{operator.__class__.__name__} must define at least one positional input parameter in __call__"
        )

    input_types: list[Any] = []
    for parameter in parameters:
        if parameter.name not in hints:
            raise PipelineValidationError(
                f"{operator.__class__.__name__} is missing a type annotation for __call__ input"
            )
        input_types.append(hints[parameter.name])
    if "return" not in hints:
        raise PipelineValidationError(
            f"{operator.__class__.__name__} is missing a return type annotation for __call__"
        )
    return tuple(input_types), hints["return"]


def is_annotation_compatible(produced: Any, expected_inputs: tuple[Any, ...]) -> bool:
    if len(expected_inputs) == 1:
        return is_single_annotation_compatible(produced, expected_inputs[0])
    produced_types = expand_output_annotation(produced)
    if len(produced_types) != len(expected_inputs):
        return False
    return all(
        is_single_annotation_compatible(produced_type, expected_type)
        for produced_type, expected_type in zip(produced_types, expected_inputs, strict=True)
    )


def is_single_annotation_compatible(produced: Any, expected: Any) -> bool:
    if expected is Any or produced is Any:
        return True
    if produced == expected:
        return True
    if is_concrete_assignable(produced, expected):
        return True
    produced_origin = get_origin(produced)
    expected_origin = get_origin(expected)
    if is_union_annotation(expected):
        return any(is_single_annotation_compatible(produced, option) for option in get_args(expected))
    if is_union_annotation(produced):
        return all(is_single_annotation_compatible(option, expected) for option in get_args(produced))
    if produced_origin is None:
        return False
    if expected_origin is None:
        return is_concrete_assignable(produced_origin, expected)
    if produced_origin != expected_origin:
        return False
    produced_args = get_args(produced)
    expected_args = get_args(expected)
    if len(produced_args) != len(expected_args):
        return False
    return all(
        is_single_annotation_compatible(produced_arg, expected_arg)
        for produced_arg, expected_arg in zip(produced_args, expected_args, strict=True)
    )


def expand_output_annotation(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin is tuple:
        return get_args(annotation)
    if isinstance(annotation, tuple):
        return annotation
    return (annotation,)


def is_concrete_assignable(produced: Any, expected: Any) -> bool:
    if not isinstance(produced, type) or not isinstance(expected, type):
        return False
    try:
        return issubclass(produced, expected)
    except TypeError:
        return False


def is_concrete(annotation: Any) -> bool:
    if annotation is None or annotation is Any:
        return False
    return all(is_concrete(arg) for arg in get_args(annotation))


def is_union_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (UnionType, getattr(__import__("typing"), "Union"))


def format_annotation(annotation: Any) -> str:
    return str(annotation).replace("typing.", "")


def format_parameter_annotations(annotations: tuple[Any, ...]) -> str:
    if len(annotations) == 1:
        return format_annotation(annotations[0])
    return "(" + ", ".join(format_annotation(annotation) for annotation in annotations) + ")"


def get_signature_target(operator: Callable[..., Any]) -> Any:
    if inspect.isfunction(operator) or inspect.ismethod(operator):
        return operator
    return getattr(operator, "__call__")
