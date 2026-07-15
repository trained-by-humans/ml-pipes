from __future__ import annotations
from collections.abc import (
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
import inspect
from types import UnionType
from typing import Any, Callable, Generic, Protocol, TypeVar, get_args, get_origin, get_type_hints

try:
    from typing import Self as _TypingSelf, get_protocol_members, is_protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Self as _TypingSelf, get_protocol_members, is_protocol

from typing_extensions import Self as _ExtensionSelf

_UNBOUND = object()
_NONE_TYPE = type(None)
_MISSING_ANNOTATION = object()
_COVARIANT = "covariant"
_INVARIANT = "invariant"
_CONTRAVARIANT = "contravariant"
_SELF_ANNOTATIONS = frozenset({_TypingSelf, _ExtensionSelf})


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


def build_union_annotation_from_options(*annotations: Any) -> Any:
    """Build a union annotation for alternative options, falling back to Any."""
    unique_annotations: list[Any] = []
    for annotation in annotations:
        if annotation is Any:
            return Any
        if annotation not in unique_annotations:
            unique_annotations.append(annotation)

    if not unique_annotations:
        return Any

    try:
        return _build_union_annotation(*unique_annotations)
    except TypeError:
        return Any


def remove_none_annotation_options(annotation: Any) -> Any | None:
    if annotation in {None, _NONE_TYPE}:
        return None
    if not is_union_annotation(annotation):
        return annotation

    remaining_options = tuple(
        option
        for option in get_args(annotation)
        if option not in {None, _NONE_TYPE}
    )
    if not remaining_options:
        return None
    return build_union_annotation_from_options(*remaining_options)


def iterable_annotation(item_annotation: Any) -> Any:
    item_annotation = _NONE_TYPE if item_annotation is None else item_annotation
    return Iterable[item_annotation]


def list_annotation(item_annotation: Any) -> Any:
    item_annotation = _NONE_TYPE if item_annotation is None else item_annotation
    return list[item_annotation]


def describe_annotation(annotation: Any) -> str:
    if annotation in {None, _NONE_TYPE}:
        return "None"
    if isinstance(annotation, type) and annotation.__module__ == "builtins":
        return annotation.__name__
    return repr(annotation)


def is_unknown_annotation(annotation: Any) -> bool:
    return annotation in {Any, object}


def is_union_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (UnionType, getattr(__import__("typing"), "Union"))


def is_iterable_annotation(annotation: Any) -> bool:
    annotation = remove_none_annotation_options(annotation)
    if annotation in {Any, object, None}:
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
    annotation = remove_none_annotation_options(annotation)
    if annotation in {Any, object, None}:
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


def is_typed_dict_annotation(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and issubclass(annotation, dict)
        and hasattr(annotation, "__annotations__")
        and hasattr(annotation, "__total__")
    )


def _resolve_annotation_owner(annotation: Any) -> type | None:
    origin = get_origin(annotation)
    if isinstance(origin, type):
        return origin
    if isinstance(annotation, type):
        return annotation
    return None


def resolve_mapping_annotation(annotation: Any) -> tuple[Any, Any] | None:
    if is_typed_dict_annotation(annotation):
        return str, Any

    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return None
    try:
        if not issubclass(owner, Mapping):
            return None
    except TypeError:
        return None

    args = get_args(annotation)
    if len(args) == 2:
        return args
    return Any, Any


def resolve_typed_dict_key_annotation(annotation: Any, key: str) -> Any:
    try:
        hints = get_type_hints(annotation)
    except Exception:
        hints = getattr(annotation, "__annotations__", {})
    return hints.get(key, _MISSING_ANNOTATION)


def resolve_iterable_item_annotation(annotation: Any) -> Any:
    annotation = remove_none_annotation_options(annotation)
    if annotation in {Any, object, None}:
        return Any
    if is_union_annotation(annotation):
        item_annotations = tuple(
            resolve_iterable_item_annotation(option)
            for option in get_args(annotation)
        )
        if not item_annotations or any(item_annotation is Any for item_annotation in item_annotations):
            return Any
        return build_union_annotation_from_options(*item_annotations)

    if annotation is str:
        return str
    if annotation in {bytes, bytearray}:
        return int

    variadic_item_annotation = variadic_tuple_item_annotation(annotation)
    if variadic_item_annotation is not None:
        return variadic_item_annotation

    shape = _annotation_shape(annotation)
    if shape is not None:
        origin, child_annotations = shape
        if origin is tuple:
            return build_union_annotation_from_options(*child_annotations)
        try:
            if issubclass(origin, Mapping):
                return child_annotations[0] if child_annotations else Any
            if issubclass(origin, Iterable):
                return child_annotations[0] if child_annotations else Any
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


def resolve_sequence_item_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {list, Sequence, MutableSequence}:
        return resolve_iterable_item_annotation(annotation)

    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION
    try:
        if issubclass(owner, Sequence) and not issubclass(owner, (str, bytes, bytearray, Mapping)):
            return Any
    except TypeError:
        return _MISSING_ANNOTATION
    return _MISSING_ANNOTATION


def is_mutable_sequence_annotation(annotation: Any) -> bool:
    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return False

    try:
        return issubclass(owner, MutableSequence)
    except TypeError:
        return False


def is_generic_indexable_annotation(annotation: Any) -> bool:
    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return False
    try:
        return not issubclass(owner, (str, bytes, bytearray, Mapping)) and hasattr(owner, "__getitem__")
    except TypeError:
        return False


def is_generic_writable_indexable_annotation(annotation: Any) -> bool:
    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return False
    try:
        return not issubclass(owner, (str, bytes, bytearray, Mapping)) and hasattr(owner, "__setitem__")
    except TypeError:
        return False


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
    return _is_assignable(source_annotation, target_annotation, set())


def _is_assignable(
    source_annotation: Any,
    target_annotation: Any,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    if isinstance(target_annotation, TypeVar):
        target_annotation = _typevar_constraint_annotation(target_annotation)
    if isinstance(source_annotation, TypeVar):
        source_annotation = _typevar_constraint_annotation(source_annotation)
    if target_annotation is Any or source_annotation is Any:
        return True
    if source_annotation == target_annotation:
        return True
    if is_union_annotation(target_annotation):
        return all(
            any(
                _is_assignable(source_option, target_option, protocol_stack)
                for target_option in get_args(target_annotation)
            )
            for source_option in _union_options(source_annotation)
        )
    if is_union_annotation(source_annotation):
        return all(
            _is_assignable(option, target_annotation, protocol_stack)
            for option in get_args(source_annotation)
        )
    return _is_non_union_assignable(
        source_annotation,
        target_annotation,
        protocol_stack,
    )


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
        if is_assignable(current_annotation, candidate_annotation):
            return current_annotation
        if is_assignable(candidate_annotation, current_annotation):
            return candidate_annotation
        return _UNBOUND

    arg_pairs = _tighten_generic_argument_pairs(current_annotation, candidate_annotation)
    if arg_pairs is None:
        return _UNBOUND

    merged_args = []
    for current_arg, candidate_arg, variance in arg_pairs:
        if variance == _INVARIANT:
            merged_arg = _try_tighten_invariant_annotation(current_arg, candidate_arg)
        elif variance == _COVARIANT:
            merged_arg = _try_tighten_covariant_annotation(current_arg, candidate_arg)
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
    return _is_concrete_assignable(source_annotation, target_annotation, set())


def _is_concrete_assignable(
    source_annotation: Any,
    target_annotation: Any,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    if not isinstance(source_annotation, type) or not isinstance(target_annotation, type):
        return False
    try:
        if issubclass(source_annotation, target_annotation):
            return True
    except TypeError:
        pass
    return _is_structurally_assignable_to_protocol(
        source_annotation,
        target_annotation,
        protocol_stack,
    )


def _is_non_union_assignable(
    source_annotation: Any,
    target_annotation: Any,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    source_shape = _annotation_shape(source_annotation)
    target_shape = _annotation_shape(target_annotation)

    if source_shape is None:
        if target_shape is None:
            return _is_concrete_assignable(
                source_annotation,
                target_annotation,
                protocol_stack,
            )
        return _is_concrete_source_assignable_to_default_generic_target(
            source_annotation,
            target_shape=target_shape,
            protocol_stack=protocol_stack,
        )

    if _is_parameterized_source_assignable_to_concrete_target(
        source_annotation,
        target_annotation,
        source_shape=source_shape,
        target_shape=target_shape,
        protocol_stack=protocol_stack,
    ):
        return True

    return _is_generic_assignable(
        source_annotation,
        target_annotation,
        source_shape=source_shape,
        target_shape=target_shape,
        protocol_stack=protocol_stack,
    )


def _is_concrete_source_assignable_to_default_generic_target(
    source_annotation: Any,
    *,
    target_shape: tuple[Any, tuple[Any, ...]],
    protocol_stack: set[tuple[type, type]],
) -> bool:
    if not isinstance(source_annotation, type):
        return False

    target_origin, target_args = target_shape
    bare_target_args = _bare_generic_args(target_origin)
    if bare_target_args is None or target_args != bare_target_args:
        return False

    return _is_concrete_assignable(source_annotation, target_origin, protocol_stack)


def _is_parameterized_source_assignable_to_concrete_target(
    source_annotation: Any,
    target_annotation: Any,
    *,
    source_shape: tuple[Any, tuple[Any, ...]] | None = None,
    target_shape: tuple[Any, tuple[Any, ...]] | None = None,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    if source_shape is None:
        source_shape = _annotation_shape(source_annotation)
    if source_shape is None:
        return False
    if target_shape is None:
        target_shape = _annotation_shape(target_annotation)
    if target_shape is not None:
        return False

    if isinstance(target_annotation, type) and is_protocol(target_annotation):
        return _is_structurally_assignable_to_protocol(
            source_annotation,
            target_annotation,
            protocol_stack,
        )

    source_origin, _ = source_shape
    return _is_concrete_assignable(source_origin, target_annotation, protocol_stack)


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


def _is_generic_assignable(
    source_annotation: Any,
    target_annotation: Any,
    *,
    source_shape: tuple[Any, tuple[Any, ...]] | None = None,
    target_shape: tuple[Any, tuple[Any, ...]] | None = None,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    arg_pairs = _generic_argument_pairs(
        source_annotation,
        target_annotation,
        source_shape=source_shape,
        target_shape=target_shape,
    )
    if arg_pairs is None:
        return False
    return all(
        _is_compatible_under_variance(
            source_arg,
            target_arg,
            variance,
            protocol_stack,
        )
        for source_arg, target_arg, variance in arg_pairs
    )


def _is_structurally_assignable_to_protocol(
    source_annotation: Any,
    target_annotation: Any,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    from ml_pipes._typing.inspection import (
        bind_method_call_parameter_names,
        method_signature_includes_receiver,
        resolve_attribute_annotation_info,
        resolve_callable_signature_annotations,
    )

    source_owner = _resolve_annotation_owner(source_annotation)
    target_owner = _resolve_annotation_owner(target_annotation)
    if not isinstance(source_owner, type) or not isinstance(target_owner, type):
        return False
    if not is_protocol(target_owner):
        return False

    try:
        source_typevar_bindings = _resolve_annotation_typevar_bindings(
            source_annotation,
        )
    except _UnboundTypevarBindingError:
        return False

    pair = (source_owner, target_owner)
    if pair in protocol_stack:
        return True

    protocol_stack.add(pair)
    try:
        for member in get_protocol_members(target_owner):
            try:
                target_member_info = resolve_attribute_annotation_info(
                    target_owner,
                    member,
                )
            except Exception:
                return False

            if target_member_info.annotation is not _MISSING_ANNOTATION:
                # TypedDict keys are mapping entries, not runtime object attributes.
                if is_typed_dict_annotation(source_annotation):
                    return False
                try:
                    source_member_info = resolve_attribute_annotation_info(
                        source_owner,
                        member,
                    )
                except Exception:
                    return False
                if source_member_info.annotation is _MISSING_ANNOTATION:
                    return False
                source_member_annotation = _specialize_source_protocol_annotation(
                    source_member_info.annotation,
                    source_owner,
                    source_typevar_bindings,
                )
                target_member_annotation = _replace_self_annotation(
                    target_member_info.annotation,
                    source_owner,
                )
                if target_member_info.is_writable:
                    if not source_member_info.is_writable:
                        return False
                    if not _is_assignable(
                        source_member_annotation,
                        target_member_annotation,
                        protocol_stack,
                    ):
                        return False
                    source_write_annotation = _specialize_source_protocol_annotation(
                        source_member_info.write_annotation,
                        source_owner,
                        source_typevar_bindings,
                    )
                    target_write_annotation = _replace_self_annotation(
                        target_member_info.write_annotation,
                        source_owner,
                    )
                    if (
                        source_write_annotation is _MISSING_ANNOTATION
                        or target_write_annotation is _MISSING_ANNOTATION
                    ):
                        return False
                    if not _is_assignable(
                        target_write_annotation,
                        source_write_annotation,
                        protocol_stack,
                    ):
                        return False
                elif not _is_assignable(
                    source_member_annotation,
                    target_member_annotation,
                    protocol_stack,
                ):
                    return False
                continue

            if not _is_protocol_method_member_assignable(
                source_owner,
                target_owner,
                member,
                protocol_stack,
                source_typevar_bindings,
                bind_method_call_parameter_names,
                method_signature_includes_receiver,
                resolve_callable_signature_annotations,
            ):
                return False
    finally:
        protocol_stack.discard(pair)

    return True


def _is_protocol_method_member_assignable(
    source_owner: type,
    target_owner: type,
    member: str,
    protocol_stack: set[tuple[type, type]],
    source_typevar_bindings: dict[TypeVar, Any],
    bind_method_call_parameter_names: Callable[..., Any],
    method_signature_includes_receiver: Callable[..., Any],
    resolve_callable_signature_annotations: Callable[..., Any],
) -> bool:
    source_member = getattr(source_owner, member, _MISSING_ANNOTATION)
    target_member = getattr(target_owner, member, _MISSING_ANNOTATION)
    if source_member is _MISSING_ANNOTATION or target_member is _MISSING_ANNOTATION:
        return False
    if not callable(source_member) or not callable(target_member):
        return False

    source_annotations = resolve_callable_signature_annotations(source_member)
    target_annotations = resolve_callable_signature_annotations(target_member)
    source_return_annotation = _specialize_source_protocol_annotation(
        source_annotations.return_annotation,
        source_owner,
        source_typevar_bindings,
    )
    target_return_annotation = _replace_self_annotation(
        target_annotations.return_annotation,
        source_owner,
    )
    if source_return_annotation is None or target_return_annotation is None:
        return False
    if not source_annotations.is_inspectable or not target_annotations.is_inspectable:
        return False

    source_includes_receiver = method_signature_includes_receiver(
        source_owner,
        member,
    )
    target_includes_receiver = method_signature_includes_receiver(
        target_owner,
        member,
    )
    if (
        source_includes_receiver is None
        or target_includes_receiver is None
    ):
        return False

    if source_includes_receiver:
        source_value_parameters = source_annotations.parameters[1:]
    else:
        source_value_parameters = source_annotations.parameters
    if target_includes_receiver:
        target_value_parameters = target_annotations.parameters[1:]
    else:
        target_value_parameters = target_annotations.parameters
    if any(
        parameter.annotation is None
        or parameter.parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in (*source_value_parameters, *target_value_parameters)
    ):
        return False

    representative_calls = _representative_protocol_method_calls(
        target_value_parameters,
    )
    if representative_calls is None:
        return False

    source_parameters_by_name = {
        parameter.parameter.name: _specialize_source_protocol_annotation(
            parameter.annotation,
            source_owner,
            source_typevar_bindings,
        )
        for parameter in source_value_parameters
    }
    target_parameters_by_name = {
        parameter.parameter.name: _replace_self_annotation(
            parameter.annotation,
            source_owner,
        )
        for parameter in target_value_parameters
    }
    for args, kwargs, placeholder_tokens in representative_calls:
        source_parameter_bindings = bind_method_call_parameter_names(
            source_member,
            args,
            kwargs,
            include_receiver=source_includes_receiver,
        )
        if source_parameter_bindings is None:
            return False

        for parameter_name, token in placeholder_tokens.items():
            source_parameter_name = source_parameter_bindings.get(token)
            source_parameter_annotation = source_parameters_by_name.get(
                source_parameter_name,
            )
            target_parameter_annotation = target_parameters_by_name.get(parameter_name)
            if (
                source_parameter_annotation is None
                or target_parameter_annotation is None
            ):
                return False
            if not _is_assignable(
                target_parameter_annotation,
                source_parameter_annotation,
                protocol_stack,
            ):
                return False

    return _is_assignable(
        _replace_self_annotation(source_return_annotation, source_owner),
        _replace_self_annotation(target_return_annotation, source_owner),
        protocol_stack,
    )


def _representative_protocol_method_calls(
    parameters: tuple[Any, ...],
) -> list[tuple[tuple[object, ...], dict[str, object], dict[str, object]]] | None:
    calls = [
        _build_protocol_method_call(parameters, include_optional=False, prefer_keywords=False),
        _build_protocol_method_call(parameters, include_optional=False, prefer_keywords=True),
        _build_protocol_method_call(parameters, include_optional=True, prefer_keywords=False),
        _build_protocol_method_call(parameters, include_optional=True, prefer_keywords=True),
    ]
    if any(call is None for call in calls):
        return None
    return [call for call in calls if call is not None]


def _build_protocol_method_call(
    parameters: tuple[Any, ...],
    *,
    include_optional: bool,
    prefer_keywords: bool,
) -> tuple[tuple[object, ...], dict[str, object], dict[str, object]] | None:
    args: list[object] = []
    kwargs: dict[str, object] = {}
    placeholder_tokens: dict[str, object] = {}

    for parameter in parameters:
        if (
            not include_optional
            and parameter.parameter.default is not inspect.Parameter.empty
        ):
            continue

        token = object()
        placeholder_tokens[parameter.parameter.name] = token
        if parameter.parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(token)
            continue
        if parameter.parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if prefer_keywords:
                kwargs[parameter.parameter.name] = token
            else:
                args.append(token)
            continue
        if parameter.parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.parameter.name] = token
            continue
        return None

    return tuple(args), kwargs, placeholder_tokens


def _replace_self_annotation(annotation: Any, concrete_owner: type) -> Any:
    return _transform_annotation(
        annotation,
        lambda part: concrete_owner if _is_self_annotation(part) else part,
    )


def _specialize_source_protocol_annotation(
    annotation: Any,
    source_owner: type,
    source_typevar_bindings: dict[TypeVar, Any],
) -> Any:
    if source_typevar_bindings:
        annotation = _apply_typevar_bindings(annotation, source_typevar_bindings)
    return _replace_self_annotation(annotation, source_owner)


def _is_self_annotation(annotation: Any) -> bool:
    return annotation in _SELF_ANNOTATIONS or (
        isinstance(annotation, TypeVar)
        and annotation.__name__ == "Self"
        and annotation.__bound__ is None
        and not annotation.__constraints__
    )


def _is_compatible_under_variance(
    source_annotation: Any,
    target_annotation: Any,
    variance: str,
    protocol_stack: set[tuple[type, type]],
) -> bool:
    if variance == _INVARIANT:
        return (
            _is_assignable(source_annotation, target_annotation, protocol_stack)
            and _is_assignable(target_annotation, source_annotation, protocol_stack)
        )
    if variance == _CONTRAVARIANT:
        return _is_assignable(target_annotation, source_annotation, protocol_stack)
    return _is_assignable(source_annotation, target_annotation, protocol_stack)


def _generic_argument_pairs(
    template_annotation,
    candidate_annotation,
    *,
    source_shape: tuple[Any, tuple[Any, ...]] | None = None,
    target_shape: tuple[Any, tuple[Any, ...]] | None = None,
) -> tuple[tuple[Any, Any, str], ...] | None:
    if source_shape is None:
        try:
            source_shape = _annotation_shape(template_annotation)
        except TypeError:
            return None
    if target_shape is None:
        try:
            target_shape = _annotation_shape(candidate_annotation)
        except TypeError:
            return None

    if source_shape is None or target_shape is None:
        return None

    source_origin, source_args = source_shape
    target_origin, target_args = target_shape

    if source_origin != target_origin and not is_concrete_assignable(source_origin, target_origin):
        return None

    adapted_source_args = source_args
    if source_origin is tuple and target_origin in {Collection, Iterable, Sequence}:
        if not source_args:
            adapted_source_args = (Any,)
        elif _is_variadic_shaped(source_args):
            adapted_source_args = (variadic_tuple_item_annotation(source_args),)
        else:
            adapted_source_args = (_build_union_annotation(*source_args),)

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


def _tighten_generic_argument_pairs(
    current_annotation: Any,
    candidate_annotation: Any,
) -> tuple[tuple[Any, Any, str], ...] | None:
    try:
        current_shape = _annotation_shape(current_annotation)
        candidate_shape = _annotation_shape(candidate_annotation)
    except TypeError:
        return None

    if current_shape is None or candidate_shape is None:
        return None

    current_origin, current_args = current_shape
    candidate_origin, candidate_args = candidate_shape

    if (
        current_origin != candidate_origin
        and not is_concrete_assignable(current_origin, candidate_origin)
    ):
        return None

    if current_origin is tuple and candidate_origin in {Collection, Iterable, Sequence}:
        candidate_item = candidate_args[0]
        if _is_variadic_shaped(current_args):
            return ((variadic_tuple_item_annotation(current_args), candidate_item, _COVARIANT),)
        return tuple(
            (current_arg, candidate_item, _COVARIANT)
            for current_arg in current_args
        )

    if candidate_origin is tuple and _is_variadic_shaped(candidate_args):
        candidate_item = variadic_tuple_item_annotation(candidate_args)
        if _is_variadic_shaped(current_args):
            return ((variadic_tuple_item_annotation(current_args), candidate_item, _COVARIANT),)
        return tuple(
            (current_arg, candidate_item, _COVARIANT)
            for current_arg in current_args
        )

    if len(current_args) != len(candidate_args):
        return None

    variances = _generic_variances(candidate_origin, candidate_args)
    return tuple(
        (current_arg, candidate_arg, variance)
        for current_arg, candidate_arg, variance in zip(
            current_args,
            candidate_args,
            variances,
            strict=True,
        )
    )


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
        return _build_union_annotation(*typevar.__constraints__)
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

    arg_pairs = _generic_argument_pairs(candidate_annotation, template_annotation)
    if arg_pairs is None:
        return {}

    return _resolve_typevar_bindings(
        tuple(template_child_annotation for _, template_child_annotation, _ in arg_pairs),
        tuple(candidate_child_annotation for candidate_child_annotation, _, _ in arg_pairs),
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


def _resolve_annotation_typevar_bindings(annotation: Any) -> dict[TypeVar, Any]:
    owner = _resolve_annotation_owner(annotation)
    if not isinstance(owner, type):
        return {}
    annotation_args = get_args(annotation)
    if not annotation_args:
        return _resolve_owner_base_typevar_bindings(owner, {})
    return _resolve_owner_typevar_bindings(owner, annotation_args)


def _resolve_owner_typevar_bindings(
    owner: type,
    annotation_args: tuple[Any, ...],
) -> dict[TypeVar, Any]:
    owner_parameters = tuple(
        parameter
        for parameter in _generic_parameters(owner)
        if isinstance(parameter, TypeVar)
    )
    if not owner_parameters:
        return {}
    if len(owner_parameters) != len(annotation_args):
        return {}

    owner_bindings = dict(zip(owner_parameters, annotation_args, strict=True))
    inherited_bindings = _resolve_owner_base_typevar_bindings(
        owner,
        owner_bindings,
    )
    return _merge_typevar_bindings(owner_bindings, inherited_bindings)


def _resolve_owner_base_typevar_bindings(
    owner: type,
    owner_bindings: dict[TypeVar, Any],
) -> dict[TypeVar, Any]:
    inherited_bindings: dict[TypeVar, Any] = {}
    for base_annotation in getattr(owner, "__orig_bases__", ()):
        base_origin = get_origin(base_annotation)
        if base_origin in {Generic, Protocol}:
            continue
        specialized_base_annotation = _apply_typevar_bindings(
            base_annotation,
            owner_bindings,
        )
        base_owner = _resolve_annotation_owner(specialized_base_annotation)
        if (
            not isinstance(base_owner, type)
            or base_owner in {Generic, Protocol}
        ):
            continue
        inherited_bindings = _merge_typevar_bindings(
            inherited_bindings,
            _resolve_annotation_typevar_bindings(specialized_base_annotation),
        )
    return inherited_bindings


def _build_union_annotation(*annotations: Any) -> Any:
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


def _try_tighten_covariant_annotation(current_annotation: Any, candidate_annotation: Any) -> Any:
    tightened = try_tighten_annotation(current_annotation, candidate_annotation)
    if tightened is not _UNBOUND:
        return tightened
    if is_assignable(current_annotation, candidate_annotation):
        return current_annotation
    if is_assignable(candidate_annotation, current_annotation):
        return candidate_annotation
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
    if origin is not None:
        origin_args = get_args(annotation)
        if origin_args:
            return origin, origin_args

        bare_generic_args = _bare_generic_args(origin)
        if bare_generic_args is not None:
            return origin, bare_generic_args

        raise ValueError(
            f"Unsupported bare generic annotation {annotation}. "
            "Use explicit type arguments."
        )

    bare_generic_args = _bare_generic_args(annotation)
    if bare_generic_args is not None:
        return annotation, bare_generic_args

    if _generic_parameters(annotation):
        raise ValueError(
            f"Unsupported bare generic annotation {annotation}. "
            "Use explicit type arguments."
        )

    return None


def _bare_generic_args(annotation: Any) -> tuple[Any, ...] | None:
    if annotation in {
        AbstractSet,
        Collection,
        Iterable,
        MutableSequence,
        MutableSet,
        Sequence,
        frozenset,
        list,
        set,
        type,
    }:
        return (Any,)
    if annotation in {Mapping, MutableMapping, dict}:
        return (Any, Any)
    if annotation is tuple:
        return (Any, Ellipsis)
    return None


def _generic_parameters(annotation: Any) -> tuple[Any, ...]:
    type_parameters = getattr(annotation, "__type_params__", ())
    if type_parameters:
        return tuple(type_parameters)

    parameters = getattr(annotation, "__parameters__", ())
    if parameters:
        return tuple(parameters)

    return ()


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
        if _bare_generic_args(annotation) is None:
            return annotation
        if annotation is tuple and len(args) == 1:
            return tuple[args[0], ...]
        return _rebuild_annotation(annotation, args)
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
