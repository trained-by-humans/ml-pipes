from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, MutableMapping, MutableSequence, MutableSet, Sequence, Set as AbstractSet
import inspect
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, TypeVar, get_args, get_origin, get_type_hints

from .context import ContextOp, Recall, Store
from .region import RegionCloser, RegionOpener

_UNBOUND = object()
_COVARIANT = "covariant"
_INVARIANT = "invariant"
_CONTRAVARIANT = "contravariant"


class PipelineValidationError(ValueError):
    pass


class StaticContractUnavailableError(Exception):
    pass


def _supports_non_call_runtime_entrypoint(operator: Any) -> bool:
    return isinstance(operator, (ContextOp, RegionOpener, RegionCloser))


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
    def effective_input_types(self) -> tuple[Any, ...]:
        return self.effective_boundary.input_types

    @property
    def effective_output_type(self) -> Any:
        return self.effective_boundary.output_type

    @staticmethod
    def _collapse_input_types(input_types: tuple[Any, ...]) -> Any:
        if len(input_types) == 1:
            return input_types[0]
        return tuple[input_types]

    @property
    def collapsed_input_type(self) -> Any:
        static_input = (
            self._collapse_input_types(self.static_boundary.input_types)
            if self.static_boundary is not None
            else None
        )
        dynamic_input = (
            self._collapse_input_types(self.dynamic_boundary.input_types)
            if self.dynamic_boundary is not None
            else None
        )

        match (static_input, dynamic_input):
            case (None, input_type):
                return input_type
            case (input_type, None):
                return input_type
            case (static_input, dynamic_input):
                return refine_input_constraint(static_input, dynamic_input)

    @property
    def collapsed_dynamic_input_type(self) -> Any | None:
        if self.dynamic_boundary is None:
            return None
        return self._collapse_input_types(self.dynamic_boundary.input_types)

    def probe_contract(self, probe_input: Any) -> tuple[tuple[Any, ...], Any] | None:
        if self.dynamic_boundary is None:
            return None

        probe_annotations = dict(self.context_inputs or {})
        try:
            return self.operator.resolve_contract(
                probe_input,
                probe_annotations,
                expand_output_annotation,
                None,
            )
        except Exception:
            return None

    def is_explicitly_transitive(self) -> bool:
        if self.dynamic_boundary is None:
            return False
        probe_input = materialize_probe_annotation(self.previous_output_type)
        if probe_input is Any:
            if self.collapsed_dynamic_input_type is None:
                return False
            probe_input = materialize_probe_annotation(self.collapsed_dynamic_input_type)
        if probe_input is Any:
            return False
        result = self.probe_contract(probe_input)
        if result is None:
            return False
        _, probe_output = result
        return can_refine_annotation(self.effective_output_type, probe_output)


class PipelineValidator:
    def __init__(self, operators: list[Any]):
        self.operators = operators

    def validate(
        self,
        pipeline_input_type: Any = Any,
        strict: bool = False,
        inference: bool = False,
    ) -> TypeContract:
        """
        Validate the pipeline under one of three boundary-tightening modes.

        Mode 1: no declared pipeline input and no backward inference.
            The pipeline starts at `Any` and can only tighten from the first
            concrete entry boundary discovered by the forward pass.

        Mode 2: declared pipeline input.
            The caller provides `pipeline_input_type`, which seeds the forward
            pass and can tighten the pipeline boundary without a backward pass.

        Mode 3: backward inference.
            When `inference=True`, a backward pass may tighten the pipeline
            boundary further if the chain remains transitive enough.

        Strict mode validates operator boundaries, not the final boundary
        tightening strategy. It therefore runs independently of whether the
        returned pipeline input came from mode 1, 2, or 3.
        """
        self._validate_regions()
        self._validate_context_interactions()
        boundaries = self._run_forward_boundary_resolution_pass(pipeline_input_type)
        self._validate_downstream_compatibility(boundaries)

        entry_input_type = boundaries[0].collapsed_input_type
        resolved_pipeline_input_type = tighten_if_more_concrete(
            pipeline_input_type,
            entry_input_type,
        )

        if inference:
            inferred_input_type = self._run_backward_input_tightening_pass(boundaries)
            resolved_pipeline_input_type = tighten_if_more_concrete(
                resolved_pipeline_input_type,
                inferred_input_type,
            )

        if strict:
            # Strict mode inspects operator boundaries themselves; it does not
            # depend on which boundary-tightening mode produced the final
            # pipeline input type returned to the caller.
            self._validate_contracts_strictly(boundaries)

        return TypeContract(
            input_type=normalize_published_annotation(resolved_pipeline_input_type),
            output_type=normalize_published_annotation(boundaries[-1].effective_output_type),
        )

    @staticmethod
    def _label_for(i: int, operator: Any) -> str:
        name = operator.__name__ if inspect.isfunction(operator) or inspect.ismethod(operator) else type(operator).__name__
        return f"{i}:{name}"

    def _validate_regions(self) -> None:
        stack: list[tuple[RegionOpener[Any, Any], int]] = []
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
        except StaticContractUnavailableError:
            return None
        return _BoundarySignature(input_types=input_types, output_type=output_type)

    def _run_forward_boundary_resolution_pass(self, pipeline_input_type: Any = Any) -> list[_OperatorBoundary]:
        boundaries: list[_OperatorBoundary] = []
        previous_output_type: Any = pipeline_input_type
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
                raise PipelineValidationError(
                    f"{operator.__class__.__name__} must define resolve_contract"
                )

            boundaries.append(
                _OperatorBoundary(
                    operator=operator,
                    previous_output_type=previous_output_type,
                    context_inputs=input_context,
                    dynamic_boundary=dynamic_boundary,
                    static_boundary=static_boundary,
                )
            )
            previous_output_type = _resolve_typevar_output(
                boundaries[-1].effective_output_type,
                previous_output_type,
                boundaries[-1].effective_input_types,
            )

        return boundaries

    def _validate_downstream_compatibility(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if is_annotation_compatible(boundary.previous_output_type, boundary.effective_input_types):
                continue

            previous_name = type(boundaries[i - 1].operator).__name__
            operator = boundary.operator
            raise PipelineValidationError(
                f"Pipeline contract mismatch at {self._label_for(i, operator)}: "
                f"{previous_name} returns {format_annotation(boundary.previous_output_type)} "
                f"but {operator.__class__.__name__} expects {format_parameter_annotations(boundary.effective_input_types)}"
            )

    @classmethod
    def _run_backward_input_tightening_pass(cls, boundaries: list[_OperatorBoundary]) -> Any:
        required_upstream_type: Any = Any
        for boundary in reversed(boundaries):
            # Example: static says `object`, dynamic says `tuple[int, Any]` -> start from `tuple[int, Any]`.
            boundary_input_type = boundary.collapsed_input_type
            # Example: downstream needs `tuple[int, str]` -> project that shape backward through this operator.
            projected_input_type = cls._project_input_back_through(boundary, required_upstream_type)
            # Example: local says `tuple[int, Any]`, projection says `tuple[int, str]` -> keep `tuple[int, str]`.
            required_upstream_type = refine_input_constraint(boundary_input_type, projected_input_type)
        return required_upstream_type

    @classmethod
    def _project_input_back_through(cls, boundary: _OperatorBoundary, inferred: Any) -> Any:
        contract_input = cls._project_contract_input_back_through(boundary, inferred)
        boundary_input_type = boundary.collapsed_input_type
        if contract_input is not Any:
            return refine_input_constraint(boundary_input_type, contract_input)
        if is_annotation_compatible(boundary.effective_output_type, (inferred,)):
            return boundary_input_type
        return Any

    @classmethod
    def _project_contract_input_back_through(cls, boundary: _OperatorBoundary, inferred: Any) -> Any:
        if boundary.dynamic_boundary is None:
            return Any

        template_input = boundary.collapsed_dynamic_input_type
        specialized_input = specialize_input_from_output_template(
            template_input,
            boundary.dynamic_boundary.output_type,
            inferred,
        )
        if specialized_input is not Any and cls._confirm_contract_projection(boundary, specialized_input, inferred):
            return specialized_input

        contract_probe = boundary.probe_contract(inferred)
        if contract_probe is None:
            return Any
        input_types, output_type = contract_probe
        collapsed_input = boundary._collapse_input_types(input_types)
        if output_type == inferred and collapsed_input == inferred:
            return inferred
        return Any

    @classmethod
    def _confirm_contract_projection(cls, boundary: _OperatorBoundary, candidate_input: Any, inferred: Any) -> bool:
        contract_probe = boundary.probe_contract(candidate_input)
        if contract_probe is None:
            return False
        _, output_type = contract_probe
        return output_type == inferred

    def _validate_contracts_strictly(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if boundary.dynamic_boundary is None and any(not is_concrete(t) for t in boundary.effective_input_types):
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: input type is unresolved (Any).\n"
                    f"  Fix: annotate the parameter with a concrete type, or implement resolve_contract "
                    f"to accept and thread the upstream type dynamically."
                )
            if not is_concrete(boundary.effective_output_type) and not boundary.is_explicitly_transitive():
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: output type is unresolved (Any).\n"
                    f"  Fix: annotate the return type with a concrete type, or implement resolve_contract "
                    f"to return the upstream type (e.g. passthrough: return (Any,), current_output)."
                )


def specialize_input_from_output_template(input_template: Any, output_template: Any, inferred_output: Any) -> Any:
    binding = bind_any_placeholder(output_template, inferred_output, None)
    if binding is _UNBOUND:
        return Any
    return replace_any_placeholder(input_template, binding)


def bind_any_placeholder(template: Any, value: Any, binding: Any) -> Any:
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
    return bind_any_placeholder_args(template_args, value_args, binding)


def bind_any_placeholder_args(template_args: tuple[Any, ...], value_args: tuple[Any, ...], binding: Any) -> Any:
    for template_arg, value_arg in zip(template_args, value_args, strict=True):
        binding = bind_any_placeholder(template_arg, value_arg, binding)
        if binding is _UNBOUND:
            return _UNBOUND
    return binding


def replace_any_placeholder(template: Any, binding: Any) -> Any:
    if template is Any:
        return Any if binding is None else binding
    origin = get_origin(template)
    if origin is None:
        return template
    args = tuple(replace_any_placeholder(arg, binding) for arg in get_args(template))
    return _rebuild_annotation(origin, args)


def refine_input_constraint(current: Any, candidate: Any) -> Any:
    merged = merge_annotations(current, candidate)
    if merged is not _UNBOUND:
        return merged
    return current


def tighten_if_more_concrete(current: Any, candidate: Any) -> Any:
    merged = merge_annotations(current, candidate)
    if merged is not _UNBOUND:
        return merged
    return current


def merge_annotations(left: Any, right: Any) -> Any:
    if left is None or left is Any:
        return right
    if right is None or right is Any:
        return left
    if isinstance(left, TypeVar):
        return _merge_typevar_annotation(left, right)
    if isinstance(right, TypeVar):
        return _merge_typevar_annotation(right, left)
    if left == right:
        return left
    if left is object:
        return right
    if right is object:
        return left
    if isinstance(left, type) and isinstance(right, type):
        try:
            if issubclass(left, right):
                return left
            if issubclass(right, left):
                return right
        except TypeError:
            return _UNBOUND
        return _UNBOUND

    left_origin = get_origin(left)
    right_origin = get_origin(right)
    if left_origin != right_origin or left_origin is None:
        return _UNBOUND

    left_args = get_args(left)
    right_args = get_args(right)
    arg_pairs = _generic_argument_pairs(left_origin, left_args, left_origin, right_args)
    if arg_pairs is None:
        return _UNBOUND

    merged_args = []
    for left_arg, right_arg, variance in arg_pairs:
        if variance == _INVARIANT:
            merged_arg = _merge_invariant_annotations(left_arg, right_arg)
        else:
            merged_arg = merge_annotations(left_arg, right_arg)
        if merged_arg is _UNBOUND:
            return _UNBOUND
        merged_args.append(merged_arg)

    return _rebuild_annotation(left_origin, tuple(merged_args))


def can_refine_annotation(current: Any, candidate: Any) -> bool:
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
    arg_pairs = _generic_argument_pairs(candidate_origin, candidate_args, current_origin, current_args)
    # Example: current=`tuple[int, Any]`, candidate=`tuple[int, str, float]` -> different arity cannot refine.
    if arg_pairs is None:
        return False

    changed = False
    for cand_arg, curr_arg, variance in arg_pairs:
        if variance == _INVARIANT:
            if _can_invariant_annotation_refine(curr_arg, cand_arg):
                changed = changed or cand_arg != curr_arg
                continue
            return False
        if curr_arg is Any:
            if cand_arg is Any:
                continue
            changed = True
            continue
        if cand_arg is Any:
            return False
        if cand_arg == curr_arg:
            continue
        if can_refine_annotation(curr_arg, cand_arg):
            changed = True
            continue
        return False
    return changed


def materialize_probe_annotation(annotation: Any) -> Any:
    if isinstance(annotation, TypeVar):
        annotation = normalize_published_annotation(annotation)
    if annotation is Any or annotation is None or annotation is object:
        return int

    origin = get_origin(annotation)
    if origin is None:
        return annotation

    args = tuple(materialize_probe_annotation(arg) for arg in get_args(annotation))
    return _rebuild_annotation(origin, args)


def get_signature_target(operator: Callable[..., Any]) -> Any:
    if inspect.isfunction(operator) or inspect.ismethod(operator):
        return operator
    try:
        return getattr(operator, "__call__")
    except AttributeError as exc:
        if _supports_non_call_runtime_entrypoint(operator):
            raise StaticContractUnavailableError from exc
        raise PipelineValidationError(
            f"{operator.__class__.__name__} must define __call__"
        ) from exc


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


def expand_output_annotation(annotation: Any) -> tuple[Any, ...]:
    origin = get_origin(annotation)
    if origin is tuple:
        return get_args(annotation)
    if isinstance(annotation, tuple):
        return annotation
    return (annotation,)


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
    if isinstance(expected, TypeVar):
        bound = _typevar_constraint_annotation(expected)
        return bound is Any or is_single_annotation_compatible(produced, bound)
    if isinstance(produced, TypeVar):
        bound = _typevar_constraint_annotation(produced)
        if bound is Any:
            return True
        return is_single_annotation_compatible(bound, expected)
    if expected is Any or produced is Any:
        return True
    if produced == expected:
        return True
    if is_concrete_assignable(produced, expected):
        return True
    if is_union_annotation(expected):
        return all(
            any(
                is_single_annotation_compatible(produced_option, expected_option)
                for expected_option in get_args(expected)
            )
            for produced_option in _union_options(produced)
        )
    produced_origin = get_origin(produced)
    expected_origin = get_origin(expected)
    if is_union_annotation(produced):
        return all(is_single_annotation_compatible(option, expected) for option in get_args(produced))
    if produced_origin is None:
        return False
    if expected_origin is None:
        return is_concrete_assignable(produced_origin, expected)
    if not _generic_origins_compatible(produced_origin, expected_origin):
        return False
    produced_args = get_args(produced)
    expected_args = get_args(expected)
    arg_pairs = _generic_argument_pairs(produced_origin, produced_args, expected_origin, expected_args)
    if arg_pairs is None:
        return False
    return all(
        _is_compatible_under_variance(produced_arg, expected_arg, variance)
        for produced_arg, expected_arg, variance in arg_pairs
    )


def _combine_annotations(*annotations: Any) -> Any:
    if not annotations:
        return Any
    combined = annotations[0]
    for annotation in annotations[1:]:
        combined = combined | annotation
    return combined


def _union_options(annotation: Any) -> tuple[Any, ...]:
    if is_union_annotation(annotation):
        return get_args(annotation)
    return (annotation,)


def _typevar_constraint_annotation(typevar: TypeVar) -> Any:
    if typevar.__bound__ is not None:
        return typevar.__bound__
    if typevar.__constraints__:
        return _combine_annotations(*typevar.__constraints__)
    return Any


def _merge_typevar_annotation(typevar: TypeVar, candidate: Any) -> Any:
    if candidate == typevar:
        return typevar
    if candidate is Any or candidate is None:
        return typevar

    constraint = _typevar_constraint_annotation(typevar)
    if constraint is Any:
        return candidate
    if is_single_annotation_compatible(candidate, constraint):
        return candidate

    normalized_constraint = normalize_published_annotation(constraint)
    merged = merge_annotations(normalized_constraint, candidate)
    if merged is not _UNBOUND:
        return merged
    return _UNBOUND


def normalize_published_annotation(annotation: Any) -> Any:
    if isinstance(annotation, TypeVar):
        return normalize_published_annotation(_typevar_constraint_annotation(annotation))

    origin = get_origin(annotation)
    if origin is None:
        return annotation

    normalized_args = tuple(normalize_published_annotation(arg) for arg in get_args(annotation))
    return _rebuild_annotation(origin, normalized_args)


def _generic_origins_compatible(produced_origin: Any, expected_origin: Any) -> bool:
    if produced_origin == expected_origin:
        return True
    return is_concrete_assignable(produced_origin, expected_origin)


def _generic_argument_pairs(
    produced_origin: Any,
    produced_args: tuple[Any, ...],
    expected_origin: Any,
    expected_args: tuple[Any, ...],
) -> tuple[tuple[Any, Any, str], ...] | None:
    adapted_produced_args = _adapt_generic_args_for_expected_origin(
        produced_origin,
        produced_args,
        expected_origin,
    )
    if expected_origin is tuple and len(expected_args) == 2 and expected_args[1] is Ellipsis:
        expected_item = expected_args[0]
        if len(adapted_produced_args) == 2 and adapted_produced_args[1] is Ellipsis:
            return ((adapted_produced_args[0], expected_item, _COVARIANT),)
        if adapted_produced_args and adapted_produced_args[-1] is Ellipsis:
            return None
        return tuple(
            (produced_arg, expected_item, _COVARIANT)
            for produced_arg in adapted_produced_args
        )

    if len(adapted_produced_args) != len(expected_args):
        return None

    variances = _generic_variances(expected_origin, expected_args)
    return tuple(
        (produced_arg, expected_arg, variance)
        for produced_arg, expected_arg, variance in zip(
            adapted_produced_args,
            expected_args,
            variances,
            strict=True,
        )
    )


def _adapt_generic_args_for_expected_origin(
    produced_origin: Any,
    produced_args: tuple[Any, ...],
    expected_origin: Any,
) -> tuple[Any, ...]:
    if produced_origin is tuple and expected_origin in {Collection, Iterable, Sequence}:
        if not produced_args:
            return (Any,)
        if len(produced_args) == 2 and produced_args[1] is Ellipsis:
            return (produced_args[0],)
        return (_combine_annotations(*produced_args),)
    return produced_args


def _generic_variances(origin: Any, args: tuple[Any, ...]) -> tuple[str, ...]:
    if origin in {AbstractSet, Collection, Iterable, Sequence, frozenset, tuple, type}:
        return (_COVARIANT,) * len(args)
    if origin is Mapping and len(args) == 2:
        return (_INVARIANT, _COVARIANT)
    return (_INVARIANT,) * len(args)


def _is_compatible_under_variance(produced: Any, expected: Any, variance: str) -> bool:
    if variance == _INVARIANT:
        return (
            is_single_annotation_compatible(produced, expected)
            and is_single_annotation_compatible(expected, produced)
        )
    if variance == _CONTRAVARIANT:
        return is_single_annotation_compatible(expected, produced)
    return is_single_annotation_compatible(produced, expected)


def _merge_invariant_annotations(left: Any, right: Any) -> Any:
    if left is None or left is Any:
        return right
    if right is None or right is Any:
        return left
    if left is object:
        return right
    if right is object:
        return left
    if left == right:
        return left
    return _UNBOUND


def _can_invariant_annotation_refine(current: Any, candidate: Any) -> bool:
    if candidate is Any or candidate is None:
        return False
    if current is Any or current is None or current is object:
        return True
    return candidate == current


def _rebuild_annotation(origin: Any, args: tuple[Any, ...]) -> Any:
    if origin is UnionType:
        rebuilt = args[0]
        for arg in args[1:]:
            rebuilt = rebuilt | arg
        return rebuilt
    if len(args) == 1:
        return origin[args[0]]
    return origin[args]


def _expand_input_annotations_for_binding(input_type: Any, input_types: tuple[Any, ...]) -> tuple[Any, ...] | None:
    if len(input_types) == 1:
        return (input_type,)
    expanded_input_type = expand_output_annotation(input_type)
    if len(expanded_input_type) != len(input_types):
        return None
    return expanded_input_type


def _bind_typevars_from_inputs(
    input_type: Any,
    input_types: tuple[Any, ...],
) -> dict[TypeVar, Any] | None:
    actual_input_types = _expand_input_annotations_for_binding(input_type, input_types)
    if actual_input_types is None:
        return None

    bindings: dict[TypeVar, Any] = {}
    for template_input, actual_input in zip(input_types, actual_input_types, strict=True):
        if not is_single_annotation_compatible(actual_input, template_input):
            return None
        if not _bind_typevars(template_input, actual_input, bindings):
            return None
    return bindings


def _bind_typevars(template: Any, value: Any, bindings: dict[TypeVar, Any]) -> bool:
    if isinstance(template, TypeVar):
        return _bind_single_typevar(template, value, bindings)

    template_origin = get_origin(template)
    if template_origin is None:
        return True

    value_origin = get_origin(value)
    if value_origin is None or not _generic_origins_compatible(value_origin, template_origin):
        return True

    template_args = get_args(template)
    value_args = get_args(value)
    arg_pairs = _generic_argument_pairs(value_origin, value_args, template_origin, template_args)
    if arg_pairs is None:
        return True

    for value_arg, template_arg, _ in arg_pairs:
        if not _bind_typevars(template_arg, value_arg, bindings):
            return False
    return True


def _bind_single_typevar(typevar: TypeVar, value: Any, bindings: dict[TypeVar, Any]) -> bool:
    if value is Any and typevar.__bound__ is not None:
        return True
    if not _value_satisfies_typevar_bound(typevar, value):
        return False

    if typevar not in bindings:
        bindings[typevar] = value
        return True

    existing = bindings[typevar]
    merged = _merge_typevar_binding(existing, value)
    if merged is _UNBOUND or not _value_satisfies_typevar_bound(typevar, merged):
        return False
    bindings[typevar] = merged
    return True


def _value_satisfies_typevar_bound(typevar: TypeVar, value: Any) -> bool:
    bound = typevar.__bound__
    if bound is None:
        return True
    return is_single_annotation_compatible(value, bound)


def _merge_typevar_binding(existing: Any, candidate: Any) -> Any:
    if existing == candidate:
        return existing
    if is_single_annotation_compatible(candidate, existing):
        return existing
    if is_single_annotation_compatible(existing, candidate):
        return candidate
    return _UNBOUND


def _replace_typevars(annotation: Any, bindings: dict[TypeVar, Any]) -> Any:
    if isinstance(annotation, TypeVar):
        return bindings.get(annotation, annotation)

    origin = get_origin(annotation)
    if origin is None:
        return annotation

    replaced_args = tuple(_replace_typevars(arg, bindings) for arg in get_args(annotation))
    return _rebuild_annotation(origin, replaced_args)


def _resolve_typevar_output(output_type: Any, input_type: Any, input_types: tuple[Any, ...]) -> Any:
    bindings = _bind_typevars_from_inputs(input_type, input_types)
    if not bindings:
        return output_type
    return _replace_typevars(output_type, bindings)


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
