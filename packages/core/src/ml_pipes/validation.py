from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ml_pipes._typing.annotation import (
    _are_annotations_equivalent,
    _collect_any_placeholder_bindings,
    _replace_any_placeholders_in_order,
    align_source_annotation_to_target_annotations,
    collapse_annotation_parts,
    expand_annotation_parts,
    format_annotation,
    format_parameter_annotations,
    is_assignable,
    is_concrete_annotation,
    is_output_annotation_assignable_to_input_annotations,
    materialize_probe_annotation,
    normalize_published_annotation,
    specialize_output_annotation_from_aligned_input_annotations,
    satisfies_annotation_constraint,
    tighten_annotation,
)
from ml_pipes._typing.inspection import resolve_callable_annotations
from ml_pipes._typing.signatures import validate_operator_signature
from ml_pipes.context import ContextOp, Recall, Store
from ml_pipes.region import RegionCloser, RegionOpener


class PipelineValidationError(ValueError):
    pass


class PipelineValidationWarning(UserWarning):
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

    @property
    def collapsed_input_annotation(self) -> Any:
        static_input_annotation = (
            collapse_annotation_parts(self.static_boundary.input_types)
            if self.static_boundary is not None
            else None
        )
        dynamic_input_annotation = (
            collapse_annotation_parts(self.dynamic_boundary.input_types)
            if self.dynamic_boundary is not None
            else None
        )

        match (static_input_annotation, dynamic_input_annotation):
            case (None, input_annotation):
                return input_annotation
            case (input_annotation, None):
                return input_annotation
            case (static_input_annotation, dynamic_input_annotation):
                return tighten_annotation(static_input_annotation, dynamic_input_annotation)

    @property
    def collapsed_dynamic_input_annotation(self) -> Any | None:
        if self.dynamic_boundary is None:
            return None
        return collapse_annotation_parts(self.dynamic_boundary.input_types)

    def probe_contract(
        self,
        probe_input: Any,
        probe_annotations: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], Any] | None:
        if self.dynamic_boundary is None:
            return None

        if probe_annotations is None:
            probe_annotations = dict(self.context_inputs or {})
        try:
            return self.operator.resolve_contract(
                probe_input,
                probe_annotations,
                PipelineValidationError,
            )
        except Exception:
            return None

    def does_contract_resolve_concretely(self) -> bool:
        if self.dynamic_boundary is None:
            return False
        probe_input = materialize_probe_annotation(self.previous_output_type)
        if probe_input is Any:
            if self.collapsed_dynamic_input_annotation is None:
                return False
            probe_input = materialize_probe_annotation(self.collapsed_dynamic_input_annotation)
        if probe_input is Any:
            return False
        probe_annotations = {
            name: materialize_probe_annotation(annotation)
            for name, annotation in (self.context_inputs or {}).items()
        }
        result = self.probe_contract(probe_input, probe_annotations)
        if result is None:
            return False
        _, probe_output = result
        return is_concrete_annotation(probe_output) and satisfies_annotation_constraint(
            self.effective_output_type,
            probe_output,
        )

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
        Validate the pipeline under one of three pipeline-input boundary modes.

        Mode 1: no declared pipeline input and no backward inference.
            The pipeline starts at `Any` and can only tighten from the entry
            boundary resolved by the forward pass. It does not back-propagate
            constraints from later steps.

        Mode 2: declared pipeline input.
            The caller provides `pipeline_input_type`, which seeds the forward
            pass, participates in compatibility checking, and can tighten the
            pipeline boundary without a backward pass.

        Mode 3: backward inference.
            When `inference=True`, a backward pass may tighten the returned
            pipeline boundary further if the chain remains transitive enough.

        Strict mode validates operator boundaries, not the final boundary
        tightening strategy. It therefore runs independently of whether the
        returned pipeline input came from mode 1, 2, or 3.
        """
        self._validate_regions()
        self._validate_context_interactions()
        boundaries = self._run_forward_boundary_resolution_pass(pipeline_input_type)
        self._validate_downstream_compatibility(boundaries)

        entry_input_annotation = boundaries[0].collapsed_input_annotation
        resolved_pipeline_input_type = tighten_annotation(
            pipeline_input_type,
            entry_input_annotation,
        )

        if inference:
            inferred_input_annotation = self._run_backward_input_tightening_pass(boundaries)
            resolved_pipeline_input_type = tighten_annotation(
                resolved_pipeline_input_type,
                inferred_input_annotation,
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
            label = self._label_for(i, op)
            match op:
                case RegionOpener() if stack and type(stack[-1][0]) is type(op):
                    parent_opener, parent_index = stack[-1]
                    parent_label = self._label_for(parent_index, parent_opener)
                    raise PipelineValidationError(
                        f"Pipeline step {label} opens a {type(op).__name__} region inside "
                        f"{parent_label}. Opening a {type(op).__name__} region inside another "
                        f"open {type(op).__name__} region is not supported."
                    )
                case RegionOpener():
                    stack.append((op, i))
                case RegionCloser() if not stack:
                    raise PipelineValidationError(
                        f"Pipeline step {label} has no matching opener"
                    )
                case RegionCloser() if not isinstance(op, stack[-1][0].closing_type):
                    top_opener, top_pos = stack[-1]
                    opener_label = self._label_for(top_pos, top_opener)
                    raise PipelineValidationError(
                        f"Pipeline step {label} is {type(op).__name__}, but the currently open "
                        f"region at {opener_label} is {type(top_opener).__name__} and must be "
                        f"closed with {top_opener.closing_type.__name__}, not {type(op).__name__}."
                    )
                case RegionCloser():
                    stack.pop()

        for opener, pos in stack:
            label = self._label_for(pos, opener)
            raise PipelineValidationError(
                f"Pipeline step {label} has no matching {opener.closing_type.__name__}"
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
                    label = self._label_for(i, operator)
                    raise PipelineValidationError(
                        f"Pipeline step {label} references a key that was not stored: "
                        f"{operator.name!r}. "
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
            PipelineValidationError,
        )
        return _BoundarySignature(input_types=input_types, output_type=output_type)

    def _resolve_static_boundary(self, i: int, operator: Any) -> _BoundarySignature | None:
        if _supports_non_call_runtime_entrypoint(operator):
            return None

        label = self._label_for(i, operator)
        parameters = validate_operator_signature(
            operator,
            label=label,
            error_type=PipelineValidationError,
            warning_type=PipelineValidationWarning,
        )
        input_types, output_type = require_operator_annotations(operator, label=label)
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

            dynamic_boundary = self._resolve_dynamic_boundary(operator, previous_output_type, stored_annotations)
            static_boundary = self._resolve_static_boundary(i, operator)
            if dynamic_boundary is None and static_boundary is None:
                label = self._label_for(i, operator)
                raise PipelineValidationError(
                    f"Pipeline step {label} must define resolve_contract"
                )

            current_boundary = _OperatorBoundary(
                operator=operator,
                previous_output_type=previous_output_type,
                context_inputs=dict(stored_annotations) if isinstance(operator, ContextOp) else None,
                dynamic_boundary=dynamic_boundary,
                static_boundary=static_boundary,
            )
            boundaries.append(current_boundary)
            aligned_candidate_annotations = align_source_annotation_to_target_annotations(
                previous_output_type,
                current_boundary.effective_input_types,
            )
            if aligned_candidate_annotations is None:
                previous_output_type = current_boundary.effective_output_type
            else:
                previous_output_type = specialize_output_annotation_from_aligned_input_annotations(
                    aligned_candidate_annotations,
                    current_boundary.effective_input_types,
                    current_boundary.effective_output_type,
                )

        return boundaries

    def _validate_downstream_compatibility(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if is_output_annotation_assignable_to_input_annotations(
                boundary.previous_output_type,
                boundary.effective_input_types,
            ):
                continue

            operator = boundary.operator
            if i == 0:
                upstream_label = "Pipeline input"
            else:
                upstream_label = type(boundaries[i - 1].operator).__name__
            raise PipelineValidationError(
                f"Pipeline contract mismatch at {self._label_for(i, operator)}: "
                f"{upstream_label} provides {format_annotation(boundary.previous_output_type)} "
                f"but {operator.__class__.__name__} expects {format_parameter_annotations(boundary.effective_input_types)}"
            )

    @classmethod
    def _run_backward_input_tightening_pass(cls, boundaries: list[_OperatorBoundary]) -> Any:
        required_downstream_type: Any = Any
        for boundary in reversed(boundaries):
            # Example: static says `object`, dynamic says `tuple[int, Any]` -> start from `tuple[int, Any]`.
            boundary_input_annotation = boundary.collapsed_input_annotation
            # Example: downstream needs `tuple[int, str]` -> project that shape backward through this operator.
            projected_input_annotation = cls._project_contract_input_back_through(
                boundary,
                required_downstream_type,
            )
            # Example: local says `tuple[int, Any]`, projection says `tuple[int, str]` -> keep `tuple[int, str]`.
            required_downstream_type = tighten_annotation(
                boundary_input_annotation,
                projected_input_annotation,
            )
        return required_downstream_type

    @classmethod
    def _project_contract_input_back_through(
        cls,
        boundary: _OperatorBoundary,
        required_output_annotation: Any,
    ) -> Any:
        if boundary.dynamic_boundary is None:
            return Any

        projected_input_annotation = cls._project_input_annotation_from_output_template(boundary, required_output_annotation)
        if projected_input_annotation is not None:
            return projected_input_annotation

        if cls._does_contract_preserve_annotation(boundary, required_output_annotation):
            return required_output_annotation
        return Any

    @classmethod
    def _project_input_annotation_from_output_template(
        cls,
        boundary: _OperatorBoundary,
        required_output_annotation: Any,
    ) -> Any | None:
        try:
            input_template = boundary.collapsed_dynamic_input_annotation
            output_template = boundary.dynamic_boundary.output_type
            placeholder_bindings = _collect_any_placeholder_bindings(output_template, required_output_annotation)
            projected_input_annotation = _replace_any_placeholders_in_order(input_template, placeholder_bindings)

            if cls._confirm_contract_projection(boundary, projected_input_annotation, required_output_annotation):
                return projected_input_annotation
        except (TypeError, ValueError) as exc:
            return None

    @classmethod
    def _confirm_contract_projection(
        cls,
        boundary: _OperatorBoundary,
        candidate_input: Any,
        expected_output_annotation: Any,
    ) -> bool:
        contract_probe = boundary.probe_contract(candidate_input)
        if contract_probe is None:
            return False
        _, output_type = contract_probe
        return _are_annotations_equivalent(output_type, expected_output_annotation)

    @classmethod
    def _does_contract_preserve_annotation(
        cls,
        boundary: _OperatorBoundary,
        annotation: Any,
    ) -> bool:
        contract_probe = boundary.probe_contract(annotation)
        if contract_probe is None:
            return False
        input_types, output_type = contract_probe
        collapsed_input_annotation = collapse_annotation_parts(input_types)
        return _are_annotations_equivalent(output_type, annotation) and _are_annotations_equivalent(
            collapsed_input_annotation,
            annotation,
        )

    def _validate_contracts_strictly(self, boundaries: list[_OperatorBoundary]) -> None:
        for i, boundary in enumerate(boundaries):
            if boundary.dynamic_boundary is None and any(
                not is_concrete_annotation(input_annotation)
                for input_annotation in boundary.effective_input_types
            ):
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: input type is unresolved (Any).\n"
                    f"  Fix: annotate the parameter with a concrete type, or implement resolve_contract "
                    f"to accept and thread the upstream type dynamically."
                )
            if not (
                is_concrete_annotation(boundary.effective_output_type)
                or boundary.does_contract_resolve_concretely()
            ):
                raise PipelineValidationError(
                    f"Strict mode violation at {self._label_for(i, boundary.operator)}: output type is unresolved (Any).\n"
                    f"  Fix: annotate the return type with a concrete type, or implement resolve_contract "
                    f"to return the upstream type (e.g. passthrough: return (Any,), current_output)."
                )

def require_operator_annotations(
    operator: Any,
    *,
    label: str,
) -> tuple[tuple[Any, ...], Any]:
    annotations = resolve_callable_annotations(operator)
    if any(annotation is None for annotation in annotations.parameter_annotations):
        raise PipelineValidationError(
            f"Pipeline step {label} is missing a type annotation for __call__ input"
        )
    if annotations.return_annotation is None:
        raise PipelineValidationError(
            f"Pipeline step {label} is missing a return type annotation for __call__"
        )
    return tuple(annotations.parameter_annotations), annotations.return_annotation
