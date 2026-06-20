from __future__ import annotations

from collections.abc import (
    Collection,
    Iterable,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from types import UnionType
from typing import Any, Callable, TypeVar, get_args, get_origin

_UNBOUND = object()
_NONE_TYPE = type(None)
_COVARIANT = "covariant"
_INVARIANT = "invariant"
_CONTRAVARIANT = "contravariant"


class _UnboundTypevarBindingError(Exception):
    pass


def expand_annotation_parts(annotation: Any) -> tuple[Any, ...]:
    """Expand an annotation into parts used for multi-parameter matching.

    Fixed tuples are expanded into one annotation per positional part.
    Variadic tuples remain atomic because Pipeline does not dispatch them as
    multi-parameter boundaries.
    """
    if _contains_ellipsize(annotation):
        if _is_variadic(annotation):
            return (annotation,)
        raise ValueError(
            f"Malformed tuple annotation with ellipsis: {annotation}. "
            "Use tuple[T, ...] for variadic tuples."
        )
    origin = get_origin(annotation)
    if origin is tuple:
        return get_args(annotation)
    if isinstance(annotation, tuple):
        return annotation
    return (annotation,)


def collapse_annotation_parts(annotation_parts: tuple[Any, ...]) -> Any:
    if len(annotation_parts) == 1:
        return annotation_parts[0]
    return tuple[annotation_parts]


def combine_annotation_options(*annotations: Any) -> Any:
    if not annotations:
        return Any

    combined = annotations[0]
    for annotation in annotations[1:]:
        if combined == annotation:
            continue
        if combined is Any or annotation is Any:
            combined = Any
            continue
        combined = _combine_annotations(combined, annotation)
    return combined


def remove_none_annotation_options_or_any(annotation: Any) -> Any:
    if annotation in {None, _NONE_TYPE}:
        return Any
    if not is_union_annotation(annotation):
        return annotation

    remaining_options = tuple(
        option
        for option in get_args(annotation)
        if option not in {None, _NONE_TYPE}
    )
    if not remaining_options:
        return Any
    return combine_annotation_options(*remaining_options)


def iterable_annotation(item_annotation: Any) -> Any:
    item_annotation = _NONE_TYPE if item_annotation is None else item_annotation
    return Iterable[item_annotation]


def list_annotation(item_annotation: Any) -> Any:
    item_annotation = _NONE_TYPE if item_annotation is None else item_annotation
    return list[item_annotation]


def is_union_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (UnionType, getattr(__import__("typing"), "Union"))


def is_iterable_annotation(annotation: Any) -> bool:
    annotation = remove_none_annotation_options_or_any(annotation)
    if annotation in {Any, object}:
        return False
    if is_union_annotation(annotation):
        options = get_args(annotation)
        return bool(options) and all(is_iterable_annotation(option) for option in options)

    if annotation in {str, bytes, bytearray}:
        return True

    origin = get_origin(annotation)
    if origin is not None:
        try:
            return issubclass(origin, Iterable)
        except TypeError:
            return False
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Iterable)
        except TypeError:
            return False
    return False


def is_mapping_annotation(annotation: Any) -> bool:
    annotation = remove_none_annotation_options_or_any(annotation)
    if annotation in {Any, object}:
        return False
    if is_union_annotation(annotation):
        options = get_args(annotation)
        return bool(options) and all(is_mapping_annotation(option) for option in options)

    origin = get_origin(annotation)
    if origin is not None:
        try:
            return issubclass(origin, Mapping)
        except TypeError:
            return False
    if isinstance(annotation, type):
        try:
            return issubclass(annotation, Mapping)
        except TypeError:
            return False
    return False


def resolve_iterable_item_annotation(annotation: Any) -> Any:
    annotation = remove_none_annotation_options_or_any(annotation)
    if annotation in {Any, object}:
        return Any
    if is_union_annotation(annotation):
        item_annotations = tuple(
            resolve_iterable_item_annotation(option)
            for option in get_args(annotation)
        )
        if not item_annotations or any(item_annotation is Any for item_annotation in item_annotations):
            return Any
        return combine_annotation_options(*item_annotations)

    if annotation is str:
        return str
    if annotation in {bytes, bytearray}:
        return int

    variadic_item_annotation = variadic_tuple_item_annotation(annotation)
    if variadic_item_annotation is not None:
        return variadic_item_annotation

    origin = get_origin(annotation)
    if origin is tuple:
        tuple_item_annotations = get_args(annotation)
        if not tuple_item_annotations:
            return Any
        return combine_annotation_options(*tuple_item_annotations)
    if origin is not None:
        generic_arguments = get_args(annotation)
        try:
            if issubclass(origin, Mapping):
                return generic_arguments[0] if generic_arguments else Any
            if issubclass(origin, Iterable):
                return generic_arguments[0] if generic_arguments else Any
        except TypeError:
            return Any
    if isinstance(annotation, type):
        try:
            if issubclass(annotation, Mapping):
                return Any
            if issubclass(annotation, Iterable):
                return Any
        except TypeError:
            return Any
    return Any


def align_source_annotation_to_target_annotations(
    source_annotation: Any,
    target_annotations: tuple[Any, ...],
) -> tuple[Any, ...] | None:
    if len(target_annotations) == 1:
        return (source_annotation,)

    aligned_source_annotations = expand_annotation_parts(source_annotation)
    if len(aligned_source_annotations) != len(target_annotations):
        return None
    return aligned_source_annotations


def is_output_annotation_assignable_to_input_annotations(
    source_output_annotation: Any,
    target_input_annotations: tuple[Any, ...],
) -> bool:
    aligned_source_annotations = align_source_annotation_to_target_annotations(
        source_output_annotation,
        target_input_annotations,
    )
    if aligned_source_annotations is None:
        return False
    return all(
        is_assignable(source_annotation, target_annotation)
        for source_annotation, target_annotation in zip(
            aligned_source_annotations,
            target_input_annotations,
            strict=True,
        )
    )


def is_assignable(
    source_annotation: Any,
    target_annotation: Any,
) -> bool:
    if isinstance(target_annotation, TypeVar):
        target_annotation = _typevar_constraint_annotation(target_annotation)
    if isinstance(source_annotation, TypeVar):
        source_annotation = _typevar_constraint_annotation(source_annotation)
    if target_annotation is Any or source_annotation is Any:
        return True
    if source_annotation == target_annotation:
        return True
    if is_concrete_assignable(source_annotation, target_annotation):
        return True
    if is_union_annotation(target_annotation):
        return all(
            any(
                is_assignable(source_option, target_option)
                for target_option in get_args(target_annotation)
            )
            for source_option in _union_options(source_annotation)
        )
    if is_union_annotation(source_annotation):
        return all(
            is_assignable(option, target_annotation)
            for option in get_args(source_annotation)
        )
    return _can_assign_generic_source_to_target_annotation(source_annotation, target_annotation)


def tighten_annotation(current_annotation: Any, candidate_annotation: Any) -> Any:
    tightened_annotation = try_tighten_annotation(current_annotation, candidate_annotation)
    if tightened_annotation is not _UNBOUND:
        return tightened_annotation
    return current_annotation


def satisfies_annotation_constraint(constraint_annotation: Any, annotation: Any) -> bool:
    return try_tighten_annotation(constraint_annotation, annotation) is not _UNBOUND


def try_tighten_annotation(current_annotation: Any, candidate_annotation: Any) -> Any:
    if current_annotation is None or current_annotation is Any:
        return candidate_annotation
    if candidate_annotation is None or candidate_annotation is Any:
        return current_annotation
    if isinstance(current_annotation, TypeVar):
        return _merge_typevar_annotation(current_annotation, candidate_annotation)
    if isinstance(candidate_annotation, TypeVar):
        return _merge_typevar_annotation(candidate_annotation, current_annotation)
    if current_annotation == candidate_annotation:
        return current_annotation
    if current_annotation is object:
        return candidate_annotation
    if candidate_annotation is object:
        return current_annotation
    if isinstance(current_annotation, type) and isinstance(candidate_annotation, type):
        try:
            if issubclass(current_annotation, candidate_annotation):
                return current_annotation
            if issubclass(candidate_annotation, current_annotation):
                return candidate_annotation
        except TypeError:
            return _UNBOUND
        return _UNBOUND

    current_shape = _annotation_shape(current_annotation)
    candidate_shape = _annotation_shape(candidate_annotation)
    if current_shape is None or candidate_shape is None:
        return _UNBOUND

    current_origin, current_args = current_shape
    candidate_origin, candidate_args = candidate_shape
    if current_origin != candidate_origin:
        return _UNBOUND

    arg_pairs = _generic_argument_pairs(
        current_origin,
        current_args,
        current_origin,
        candidate_args,
    )
    if arg_pairs is None:
        return _UNBOUND

    merged_args = []
    for current_arg, candidate_arg, variance in arg_pairs:
        if variance == _INVARIANT:
            merged_arg = _try_tighten_invariant_annotation(current_arg, candidate_arg)
        else:
            merged_arg = try_tighten_annotation(current_arg, candidate_arg)
        if merged_arg is _UNBOUND:
            return _UNBOUND
        merged_args.append(merged_arg)

    return _rebuild_annotation_like(current_annotation, tuple(merged_args))


def normalize_published_annotation(annotation: Any) -> Any:
    return _transform_annotation(annotation, _normalize_published_annotation)


def materialize_probe_annotation(annotation: Any) -> Any:
    return _transform_annotation(annotation, _materialize_probe_annotation)


def is_concrete_assignable(source_annotation: Any, target_annotation: Any) -> bool:
    if not isinstance(source_annotation, type) or not isinstance(target_annotation, type):
        return False
    try:
        return issubclass(source_annotation, target_annotation)
    except TypeError:
        return False


def is_concrete_annotation(annotation: Any) -> bool:
    if annotation is None or annotation is Any:
        return False
    shape = _annotation_shape(annotation)
    if shape is None:
        return True
    _, child_annotations = shape
    return all(is_concrete_annotation(child_annotation) for child_annotation in child_annotations)


def format_annotation(annotation: Any) -> str:
    return str(annotation).replace("typing.", "")


def format_parameter_annotations(annotations: tuple[Any, ...]) -> str:
    if len(annotations) == 1:
        return format_annotation(annotations[0])
    return "(" + ", ".join(format_annotation(annotation) for annotation in annotations) + ")"


def specialize_output_annotation_from_aligned_input_annotations(
    aligned_candidate_annotations: tuple[Any, ...],
    input_template_annotations: tuple[Any, ...],
    output_template_annotation: Any,
) -> Any:
    try:
        bindings = _resolve_typevar_bindings(
            input_template_annotations,
            aligned_candidate_annotations,
        )
        return _apply_typevar_bindings(output_template_annotation, bindings)
    except _UnboundTypevarBindingError:
        return output_template_annotation


def _are_annotations_equivalent(left_annotation: Any, right_annotation: Any) -> bool:
    if left_annotation == right_annotation:
        return True
    left_shape = _annotation_shape(left_annotation)
    right_shape = _annotation_shape(right_annotation)
    if left_shape is None or right_shape is None:
        return False

    left_origin, left_child_annotations = left_shape
    right_origin, right_child_annotations = right_shape
    if left_origin != right_origin or len(left_child_annotations) != len(right_child_annotations):
        return False

    return all(
        _are_annotations_equivalent(left_child_annotation, right_child_annotation)
        for left_child_annotation, right_child_annotation in zip(
            left_child_annotations,
            right_child_annotations,
            strict=True,
        )
    )


def _all_annotations_equivalent(annotations: list[Any]) -> bool:
    if len(annotations) < 2:
        return True
    first_annotation = annotations[0]
    return all(
        _are_annotations_equivalent(first_annotation, annotation)
        for annotation in annotations[1:]
    )


def _can_assign_generic_source_to_target_annotation(source_annotation: Any, target_annotation: Any) -> bool:
    source_shape = _annotation_shape(source_annotation)
    if source_shape is None:
        return False
    target_shape = _annotation_shape(target_annotation)
    source_origin, source_args = source_shape
    if target_shape is None:
        return is_concrete_assignable(source_origin, target_annotation)
    target_origin, target_args = target_shape
    if source_origin != target_origin and not is_concrete_assignable(source_origin, target_origin):
        return False

    arg_pairs = _generic_argument_pairs(source_origin, source_args, target_origin, target_args)
    if arg_pairs is None:
        return False
    return all(
        _is_compatible_under_variance(source_arg, target_arg, variance)
        for source_arg, target_arg, variance in arg_pairs
    )


def _is_compatible_under_variance(source_annotation: Any, target_annotation: Any, variance: str) -> bool:
    if variance == _INVARIANT:
        return (
            is_assignable(source_annotation, target_annotation)
            and is_assignable(target_annotation, source_annotation)
        )
    if variance == _CONTRAVARIANT:
        return is_assignable(target_annotation, source_annotation)
    return is_assignable(source_annotation, target_annotation)


def _generic_argument_pairs(
    source_origin: Any,
    source_args: tuple[Any, ...],
    target_origin: Any,
    target_args: tuple[Any, ...],
) -> tuple[tuple[Any, Any, str], ...] | None:
    adapted_source_args = _adapt_generic_args_for_target_origin(
        source_origin,
        source_args,
        target_origin,
    )
    if target_origin is tuple and _is_variadic_shaped(target_args):
        target_item = variadic_tuple_item_annotation(target_args)
        if _is_variadic_shaped(adapted_source_args):
            return ((variadic_tuple_item_annotation(adapted_source_args), target_item, _COVARIANT),)
        return tuple(
            (source_arg, target_item, _COVARIANT)
            for source_arg in adapted_source_args
        )

    if len(adapted_source_args) != len(target_args):
        return None

    variances = _generic_variances(target_origin, target_args)
    return tuple(
        (source_arg, target_arg, variance)
        for source_arg, target_arg, variance in zip(
            adapted_source_args,
            target_args,
            variances,
            strict=True,
        )
    )


def _adapt_generic_args_for_target_origin(
    source_origin: Any,
    source_args: tuple[Any, ...],
    target_origin: Any,
) -> tuple[Any, ...]:
    if source_origin is tuple and target_origin in {Collection, Iterable, Sequence}:
        if not source_args:
            return (Any,)
        if _is_variadic_shaped(source_args):
            return (variadic_tuple_item_annotation(source_args),)
        return (_combine_annotations(*source_args),)
    return source_args


def _generic_variances(origin: Any, args: tuple[Any, ...]) -> tuple[str, ...]:
    if origin in {AbstractSet, Collection, Iterable, Sequence, frozenset, tuple, type}:
        return (_COVARIANT,) * len(args)
    if origin is Mapping and len(args) == 2:
        return (_INVARIANT, _COVARIANT)
    return (_INVARIANT,) * len(args)


def _typevar_constraint_annotation(typevar: TypeVar) -> Any:
    if typevar.__bound__ is not None:
        return typevar.__bound__
    if typevar.__constraints__:
        return _combine_annotations(*typevar.__constraints__)
    return Any


def _resolve_typevar_bindings(
    template_annotations: tuple[Any, ...],
    candidate_annotations: tuple[Any, ...],
) -> dict[TypeVar, Any]:
    bindings: dict[TypeVar, Any] = {}
    for template_annotation, candidate_annotation in zip(
        template_annotations,
        candidate_annotations,
        strict=True,
    ):
        if not is_assignable(candidate_annotation, template_annotation):
            raise _UnboundTypevarBindingError

        pair_bindings = _resolve_typevar_bindings_from_match(
            template_annotation,
            candidate_annotation,
        )
        bindings = _merge_typevar_bindings(bindings, pair_bindings)
    return bindings


def _resolve_typevar_bindings_from_match(
    template_annotation: Any,
    candidate_annotation: Any,
) -> dict[TypeVar, Any]:
    if isinstance(template_annotation, TypeVar):
        if candidate_annotation is Any and _typevar_constraint_annotation(template_annotation) is not Any:
            return {}
        return {template_annotation: candidate_annotation}

    template_shape = _annotation_shape(template_annotation)
    if template_shape is None:
        return {}

    candidate_shape = _annotation_shape(candidate_annotation)
    if candidate_shape is None:
        return {}

    template_origin, template_child_annotations = template_shape
    candidate_origin, candidate_child_annotations = candidate_shape
    if candidate_origin != template_origin and not is_concrete_assignable(candidate_origin, template_origin):
        return {}

    child_annotation_pairs = _generic_argument_pairs(
        candidate_origin,
        candidate_child_annotations,
        template_origin,
        template_child_annotations,
    )
    if child_annotation_pairs is None:
        return {}

    return _resolve_typevar_bindings(
        tuple(template_child_annotation for _, template_child_annotation, _ in child_annotation_pairs),
        tuple(candidate_child_annotation for candidate_child_annotation, _, _ in child_annotation_pairs),
    )


def _merge_typevar_bindings(
    current_bindings: dict[TypeVar, Any],
    candidate_bindings: dict[TypeVar, Any],
) -> dict[TypeVar, Any]:
    if not candidate_bindings:
        return current_bindings

    merged_bindings = dict(current_bindings)
    for typevar, candidate_binding in candidate_bindings.items():
        if typevar not in merged_bindings:
            merged_binding = candidate_binding
        else:
            merged_binding = _tighten_typevar_binding(
                merged_bindings[typevar],
                candidate_binding,
            )
        merged_bindings[typevar] = merged_binding
    return merged_bindings


def _tighten_typevar_binding(
    current_binding: Any,
    candidate_binding: Any,
) -> Any:
    if current_binding == candidate_binding:
        tightened_binding = current_binding
    elif is_assignable(candidate_binding, current_binding):
        tightened_binding = current_binding
    elif is_assignable(current_binding, candidate_binding):
        tightened_binding = candidate_binding
    else:
        raise _UnboundTypevarBindingError

    return tightened_binding


def _apply_typevar_bindings(
    template_annotation: Any,
    bindings: dict[TypeVar, Any],
) -> Any:
    return _transform_annotation(
        template_annotation,
        lambda template_part: bindings.get(template_part, template_part)
        if isinstance(template_part, TypeVar)
        else template_part,
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


def _merge_typevar_annotation(typevar: TypeVar, candidate: Any) -> Any:
    if candidate == typevar:
        return typevar
    if candidate is Any or candidate is None:
        return typevar

    constraint = _typevar_constraint_annotation(typevar)
    if constraint is Any:
        return candidate
    if is_assignable(candidate, constraint):
        return candidate

    normalized_constraint = normalize_published_annotation(constraint)
    merged = try_tighten_annotation(normalized_constraint, candidate)
    if merged is not _UNBOUND:
        return merged
    return _UNBOUND


def _try_tighten_invariant_annotation(current_annotation: Any, candidate_annotation: Any) -> Any:
    if current_annotation is None or current_annotation is Any:
        return candidate_annotation
    if candidate_annotation is None or candidate_annotation is Any:
        return current_annotation
    if current_annotation is object:
        return candidate_annotation
    if candidate_annotation is object:
        return current_annotation
    if current_annotation == candidate_annotation:
        return current_annotation
    return _UNBOUND


def _normalize_published_annotation(annotation: Any) -> Any:
    if not isinstance(annotation, TypeVar):
        return annotation
    return normalize_published_annotation(_typevar_constraint_annotation(annotation))


def _materialize_probe_annotation(annotation: Any) -> Any:
    if isinstance(annotation, TypeVar):
        return materialize_probe_annotation(
            normalize_published_annotation(_typevar_constraint_annotation(annotation))
        )
    if annotation is Any or annotation is None or annotation is object:
        return int
    return annotation


def _collect_any_placeholder_bindings(
    template_annotation: Any,
    value_annotation: Any,
) -> list[Any] | None:
    bindings: list[Any] = []
    if not _collect_any_placeholder_bindings_into(template_annotation, value_annotation, bindings):
        return None
    return bindings


def _collect_any_placeholder_bindings_into(
    template_annotation: Any,
    value_annotation: Any,
    bindings: list[Any],
) -> bool:
    if template_annotation is Any:
        bindings.append(value_annotation)
        return True
    if template_annotation == value_annotation:
        return True

    template_shape = _annotation_shape(template_annotation)
    value_shape = _annotation_shape(value_annotation)
    if template_shape is None or value_shape is None:
        return False
    template_origin, template_children = template_shape
    value_origin, value_children = value_shape
    if template_origin != value_origin:
        return False
    if len(template_children) != len(value_children):
        return False

    return all(
        _collect_any_placeholder_bindings_into(template_child, value_child, bindings)
        for template_child, value_child in zip(template_children, value_children, strict=True)
    )


def _replace_any_placeholders_in_order(
    template_annotation: Any,
    bindings: list[Any],
) -> Any | None:
    replacement = _replace_any_placeholders_in_order_recursive(
        template_annotation,
        bindings,
        0,
    )
    if replacement is None:
        return None
    replaced_annotation, consumed_bindings = replacement
    if consumed_bindings != len(bindings) and not _all_annotations_equivalent(bindings):
        return None
    return replaced_annotation


def _replace_any_placeholders_in_order_recursive(
    template_annotation: Any,
    bindings: list[Any],
    binding_index: int,
) -> tuple[Any, int] | None:
    if template_annotation is Any:
        if binding_index >= len(bindings):
            return None
        return bindings[binding_index], binding_index + 1

    template_shape = _annotation_shape(template_annotation)
    if template_shape is None:
        return template_annotation, binding_index

    _, template_children = template_shape
    replaced_children = []
    for template_child in template_children:
        replacement = _replace_any_placeholders_in_order_recursive(
            template_child,
            bindings,
            binding_index,
        )
        if replacement is None:
            return None
        replaced_child, binding_index = replacement
        replaced_children.append(replaced_child)
    return _rebuild_annotation_like(template_annotation, tuple(replaced_children)), binding_index


def _annotation_shape(annotation: Any) -> tuple[Any, tuple[Any, ...]] | None:
    if _contains_ellipsize(annotation):
        if not _is_variadic(annotation):
            raise ValueError(
                f"Malformed tuple annotation with ellipsis: {annotation}. "
                "Use tuple[T, ...] for variadic tuples."
            )

    if isinstance(annotation, tuple):
        return tuple, annotation

    origin = get_origin(annotation)
    if origin is None:
        return None
    return origin, get_args(annotation)


def _contains_ellipsize(annotation: Any) -> bool:
    if isinstance(annotation, tuple):
        return any(part is Ellipsis for part in annotation)
    origin = get_origin(annotation)
    return origin is tuple and any(part is Ellipsis for part in get_args(annotation))


def _is_variadic(annotation: Any) -> bool:
    if isinstance(annotation, tuple):
        return _is_variadic_shaped(annotation)
    return _is_variadic_tuple(annotation)


def _is_variadic_shaped(annotation: Any) -> bool:
    if not isinstance(annotation, tuple):
        return False
    return len(annotation) == 2 and annotation[1] is Ellipsis


def _is_variadic_tuple(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is not tuple:
        return False
    return _is_variadic_shaped(get_args(annotation))


def variadic_tuple_item_annotation(annotation: Any) -> Any | None:
    if isinstance(annotation, tuple):
        if not _is_variadic_shaped(annotation):
            return None
        return annotation[0]
    if not _is_variadic_tuple(annotation):
        return None
    return get_args(annotation)[0]


def _transform_annotation(annotation: Any, transform: Callable[[Any], Any]) -> Any:
    shape = _annotation_shape(annotation)
    if shape is None:
        return transform(annotation)
    _, args = shape
    rebuilt = _rebuild_annotation_like(
        annotation,
        tuple(_transform_annotation(arg, transform) for arg in args),
    )
    return transform(rebuilt)


def _rebuild_annotation_like(annotation: Any, args: tuple[Any, ...]) -> Any:
    if isinstance(annotation, tuple):
        return tuple(args)
    rebuilt_variadic_tuple = _rebuild_variadic_tuple_annotation(annotation, args)
    if rebuilt_variadic_tuple is not None:
        return rebuilt_variadic_tuple
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    return _rebuild_annotation(origin, args)


def _rebuild_variadic_tuple_annotation(annotation: Any, args: tuple[Any, ...]) -> Any | None:
    if not _is_variadic_tuple(annotation):
        return None
    if len(args) == 1:
        return tuple[args[0], ...]
    if _is_variadic_shaped(args):
        return tuple[variadic_tuple_item_annotation(args), ...]
    return tuple[args]


def _rebuild_annotation(origin: Any, args: tuple[Any, ...]) -> Any:
    if origin is UnionType:
        rebuilt = args[0]
        for arg in args[1:]:
            rebuilt = rebuilt | arg
        return rebuilt
    if len(args) == 1:
        return origin[args[0]]
    return origin[args]
