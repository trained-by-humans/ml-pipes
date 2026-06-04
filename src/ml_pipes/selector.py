from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from types import UnionType
from typing import Any, NoReturn, TypeAlias, Union, get_args, get_origin, get_type_hints


SelectorPart = str | int
SelectorInput: TypeAlias = SelectorPart | tuple["SelectorInput", ...]
_NONE_TYPE = type(None)
_MISSING = object()
_MISSING_ANNOTATION = object()


class SelectorRuntimeError(TypeError):
    def __init__(self, message: str, *, selector: Selector | None = None, path: str | None = None) -> None:
        self.selector = selector
        self.path = path
        super().__init__(message)


class SelectorIndexError(IndexError):
    def __init__(self, message: str, *, selector: Selector | None = None, path: str | None = None) -> None:
        self.selector = selector
        self.path = path
        super().__init__(message)


class _SelectorValidationFallback(Exception):
    pass


@dataclass(frozen=True, repr=False)
class Selector:
    steps: tuple[SelectorPart, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self) -> str:
        return repr(self.steps)

    @classmethod
    def from_input(cls, selector: SelectorInput | None) -> Selector:
        if selector is None:
            return cls(())
        collapsed = _collapse_selector_input(selector)
        selector_parts = _normalize_selector_parts(collapsed)
        return cls(selector_parts)

    def render_path(self, root_label: str = "x", upto: int | None = None) -> str:
        parts = self.steps if upto is None else self.steps[: max(0, upto)]
        rendered = root_label
        for part in parts:
            if isinstance(part, int):
                rendered = f"{rendered}[{part}]" if rendered else f"[{part}]"
            else:
                rendered = f"{rendered}.{part}" if rendered else part
        return rendered

    def select_value(
        self,
        value: Any,
        *,
        error_prefix: str | None = None,
        root_label: str = "x",
    ) -> Any:
        current = value
        for step_index, part in enumerate(self.steps):
            current = _runtime_read_step(
                current,
                part,
                self,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        return current

    def select_value_or_missing(
        self,
        value: Any,
        *,
        missing: Any = _MISSING,
        root_label: str = "x",
    ) -> Any:
        try:
            return self.select_value(value, root_label=root_label)
        except (SelectorRuntimeError, SelectorIndexError):
            return missing

    def select_field(
        self,
        value: Any,
        *,
        root_label: str = "x",
    ) -> SelectedField:
        if not self.steps:
            raise ValueError("Selector.select_field requires a non-empty selector")

        return SelectedField(
            root=value,
            parent_selector=Selector(self.steps[:-1]),
            field_selection=self.steps[-1],
            parent_path=self.render_path(root_label, upto=len(self.steps) - 1),
            field_path=self.render_path(root_label),
            root_label=root_label,
        )

    def validate_read(
        self,
        annotation: Any,
        *,
        validation_error_type: type[Exception] | None = TypeError,
        error_prefix: str | None = None,
        root_label: str = "x",
    ) -> Any:
        try:
            current = annotation
            for step_index, part in enumerate(self.steps):
                current = _validate_read_step(
                    current,
                    part,
                    self,
                    step_index,
                    validation_error_type=validation_error_type,
                    error_prefix=error_prefix,
                    root_label=root_label,
                )
            return current
        except _SelectorValidationFallback:
            return Any

    def validate_write(
        self,
        annotation: Any,
        *,
        validation_error_type: type[Exception] | None = TypeError,
        error_prefix: str | None = None,
        root_label: str = "x",
    ) -> Any:
        if not self.steps:
            raise ValueError("Selector.validate_write requires a non-empty selector")

        try:
            current = annotation
            for step_index, part in enumerate(self.steps[:-1]):
                current = _validate_read_step(
                    current,
                    part,
                    self,
                    step_index,
                    validation_error_type=validation_error_type,
                    error_prefix=error_prefix,
                    root_label=root_label,
                )
            return _validate_write_target(
                current,
                self.steps[-1],
                self,
                len(self.steps) - 1,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        except _SelectorValidationFallback:
            return Any


@dataclass(frozen=True)
class SelectedField:
    root: object
    parent_selector: Selector
    field_selection: SelectorPart
    parent_path: str
    field_path: str
    root_label: str = "x"

    def set(
        self,
        value: Any,
        *,
        create_missing_mappings: bool = False,
        error_prefix: str | None = None,
    ) -> None:
        parent = self._select_parent(
            create_missing_mappings=create_missing_mappings,
            error_prefix=error_prefix,
        )
        leaf_selector = Selector((self.field_selection,))
        try:
            _runtime_write_step(
                parent,
                self.field_selection,
                value,
                leaf_selector,
                0,
                error_prefix=None,
                root_label=self.parent_path,
            )
        except (SelectorRuntimeError, SelectorIndexError) as exc:
            raise _wrap_selected_field_error(
                exc,
                action="write",
                parent_path=self.parent_path,
                field_path=self.field_path,
                error_prefix=error_prefix,
            ) from exc

    def _select_parent(
        self,
        *,
        create_missing_mappings: bool,
        error_prefix: str | None,
    ) -> Any:
        current = self.root
        for step_index, part in enumerate(self.parent_selector.steps):
            current = _runtime_parent_step(
                current,
                part,
                self.parent_selector,
                step_index,
                create_missing_mappings=create_missing_mappings,
                error_prefix=error_prefix,
                root_label=self.root_label,
            )
        return current

# Parsing

def _coerce_string_step(token: str) -> SelectorPart:
    if token.startswith("-") and token[1:].isdigit():
        return int(token)
    if token.isdigit():
        return int(token)
    return token


def _expand_selector_parts(part: SelectorPart) -> tuple[SelectorPart, ...]:
    if isinstance(part, int):
        return (part,)

    if part == "":
        raise ValueError("Selector cannot be empty")

    parts: list[SelectorPart] = []
    index = 0
    length = len(part)
    while index < length:
        current = part[index]
        if current == ".":
            raise ValueError(f"Selector {part!r} cannot contain empty path segments")

        if current == "[":
            end = part.find("]", index + 1)
            if end < 0:
                raise ValueError(f"Selector {part!r} has an unmatched '['")
            token = part[index + 1:end]
            if token == "":
                raise ValueError(f"Selector {part!r} has an empty index step")
            step = _coerce_string_step(token)
            if not isinstance(step, int):
                raise ValueError(f"Selector {part!r} has a non-integer bracket step {token!r}")
            parts.append(step)
            index = end + 1
            if index < length and part[index] not in ".[":
                raise ValueError(f"Selector {part!r} must separate steps with '.'")
            if index < length and part[index] == ".":
                index += 1
            continue

        start = index
        while index < length and part[index] not in ".[":
            index += 1
        token = part[start:index]
        if token == "":
            raise ValueError(f"Selector {part!r} cannot contain empty path segments")
        parts.append(_coerce_string_step(token))
        if index < length and part[index] == ".":
            index += 1

    if part.endswith("."):
        raise ValueError(f"Selector {part!r} cannot end with '.'")

    return tuple(parts)


def _collapse_selector_input(selector: SelectorInput | None) -> tuple[SelectorPart, ...]:
    if selector is None:
        return ()
    if isinstance(selector, (int, str)):
        return (selector,)
    if isinstance(selector, tuple):
        parts: list[SelectorPart] = []
        for item in selector:
            parts.extend(_collapse_selector_input(item))
        return tuple(parts)
    raise TypeError(f"Unsupported selector value: {selector!r}")


def _normalize_selector_parts(parts: tuple[SelectorPart, ...]) -> tuple[SelectorPart, ...]:
    normalized: list[SelectorPart] = []
    for part in parts:
        normalized.extend(_expand_selector_parts(part))
    return tuple(normalized)


# Selection


def _prefixed_message(error_prefix: str | None, message: str) -> str:
    if error_prefix:
        return f"{error_prefix} {message}"
    return message


def _is_string_like(value: object) -> bool:
    return isinstance(value, (str, bytes, bytearray))


def _supports_runtime_indexing(value: object) -> bool:
    return not isinstance(value, Mapping) and not _is_string_like(value) and hasattr(value, "__getitem__")


def _supports_runtime_assignment(value: object) -> bool:
    return not isinstance(value, Mapping) and not _is_string_like(value) and hasattr(value, "__setitem__")


def _raise_runtime_error(
    selector: Selector,
    step_index: int,
    reason: str,
    *,
    error_prefix: str | None,
    root_label: str,
) -> NoReturn:
    path = selector.render_path(root_label, upto=step_index + 1)
    raise SelectorRuntimeError(
        _prefixed_message(error_prefix, f"cannot resolve {path}: {reason}"),
        selector=selector,
        path=path,
    )


def _raise_runtime_index_error(
    selector: Selector,
    step_index: int,
    reason: str,
    *,
    error_prefix: str | None,
    root_label: str,
) -> NoReturn:
    path = selector.render_path(root_label, upto=step_index + 1)
    raise SelectorIndexError(
        _prefixed_message(error_prefix, f"is out of bounds at {path}: {reason}"),
        selector=selector,
        path=path,
    )


def _runtime_read_step(
    current: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    match part, current:
        case str() as key, Mapping() as mapping:
            return _runtime_read_mapping_key(
                mapping,
                key,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case str() as attribute, value:
            return _runtime_read_attribute(
                value,
                attribute,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as key, Mapping() as mapping:
            return _runtime_read_mapping_key(
                mapping,
                key,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as index, value if _supports_runtime_indexing(value):
            return _runtime_read_index(
                value,
                index,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as index, value:
            current_path = selector.render_path(root_label, upto=step_index)
            _raise_runtime_error(
                selector,
                step_index,
                f"{type(value).__name__} at {current_path} is not indexable",
                error_prefix=error_prefix,
                root_label=root_label,
            )
    raise AssertionError(f"Unsupported selector part: {part!r}")


def _runtime_parent_step(
    current: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    create_missing_mappings: bool,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    try:
        return _runtime_read_step(
            current,
            part,
            selector,
            step_index,
            error_prefix=error_prefix,
            root_label=root_label,
        )
    except SelectorRuntimeError:
        if create_missing_mappings and isinstance(current, MutableMapping):
            current[part] = {}
            return current[part]
        raise


def _runtime_write_step(
    parent: Any,
    part: SelectorPart,
    value: Any,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> None:
    match part, parent:
        case str() as key, MutableMapping() as mapping:
            _runtime_write_mapping_key(mapping, key, value)
            return
        case str() as attribute, target:
            _runtime_write_attribute(
                target,
                attribute,
                value,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
            return
        case int() as key, MutableMapping() as mapping:
            _runtime_write_mapping_key(mapping, key, value)
            return
        case int() as index, target if _supports_runtime_assignment(target):
            _runtime_write_index(
                target,
                index,
                value,
                selector,
                step_index,
                error_prefix=error_prefix,
                root_label=root_label,
            )
            return
        case int() as index, target:
            current_path = selector.render_path(root_label, upto=step_index)
            _raise_runtime_error(
                selector,
                step_index,
                f"{type(target).__name__} at {current_path} does not support item assignment",
                error_prefix=error_prefix,
                root_label=root_label,
            )
    raise AssertionError(f"Unsupported selector part: {part!r}")


def _wrap_selected_field_error(
    exc: SelectorRuntimeError | SelectorIndexError,
    *,
    action: str,
    parent_path: str,
    field_path: str,
    error_prefix: str | None,
) -> SelectorRuntimeError | SelectorIndexError:
    return type(exc)(
        _prefixed_message(error_prefix, f"{action} failed in parent at {parent_path}: {exc}"),
        selector=exc.selector,
        path=field_path,
    )


def _runtime_read_mapping_key(
    current: Mapping[Any, Any],
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    if part in current:
        return current[part]
    _raise_runtime_error(
        selector,
        step_index,
        f"{type(current).__name__} at {current_path} has no key {part!r}",
        error_prefix=error_prefix,
        root_label=root_label,
    )


def _runtime_read_attribute(
    current: Any,
    part: str,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    try:
        return getattr(current, part)
    except AttributeError:
        if _supports_runtime_indexing(current):
            _raise_runtime_error(
                selector,
                step_index,
                f"{type(current).__name__} at {current_path} requires an integer index, got {part!r}",
                error_prefix=error_prefix,
                root_label=root_label,
            )
        _raise_runtime_error(
            selector,
            step_index,
            f"{type(current).__name__} at {current_path} has no attribute {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )


def _runtime_read_index(
    current: Any,
    part: int,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    try:
        return current[part]
    except IndexError:
        length = len(current) if hasattr(current, "__len__") else "unknown"
        _raise_runtime_index_error(
            selector,
            step_index,
            f"{type(current).__name__} at {current_path} has length {length}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    except TypeError:
        _raise_runtime_error(
            selector,
            step_index,
            f"{type(current).__name__} at {current_path} cannot be indexed by {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    except KeyError:
        _raise_runtime_error(
            selector,
            step_index,
            f"{type(current).__name__} at {current_path} has no key {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )


def _runtime_write_mapping_key(
    parent: MutableMapping[Any, Any],
    part: SelectorPart,
    value: Any,
) -> None:
    parent[part] = value


def _runtime_write_attribute(
    parent: Any,
    part: str,
    value: Any,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> None:
    current_path = selector.render_path(root_label, upto=step_index)
    try:
        setattr(parent, part, value)
    except (AttributeError, TypeError):
        _raise_runtime_error(
            selector,
            step_index,
            f"{type(parent).__name__} at {current_path} has no writable attribute {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )


def _runtime_write_index(
    parent: Any,
    part: int,
    value: Any,
    selector: Selector,
    step_index: int,
    *,
    error_prefix: str | None,
    root_label: str,
) -> None:
    current_path = selector.render_path(root_label, upto=step_index)
    try:
        parent[part] = value
    except IndexError:
        length = len(parent) if hasattr(parent, "__len__") else "unknown"
        _raise_runtime_index_error(
            selector,
            step_index,
            f"{type(parent).__name__} at {current_path} has length {length}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    except TypeError:
        _raise_runtime_error(
            selector,
            step_index,
            f"{type(parent).__name__} at {current_path} cannot assign index {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )


# Validation


def _is_union_annotation(annotation: Any) -> bool:
    return get_origin(annotation) in {UnionType, Union}


def _annotation_description(annotation: Any) -> str:
    if annotation is None or annotation is _NONE_TYPE:
        return "None"
    if isinstance(annotation, type) and annotation.__module__ == "builtins":
        return annotation.__name__
    return repr(annotation)


def _combine_annotations(*annotations: Any) -> Any:
    unique: list[Any] = []
    for annotation in annotations:
        if annotation is Any:
            return Any
        if annotation not in unique:
            unique.append(annotation)
    if not unique:
        return Any
    combined = unique[0]
    for annotation in unique[1:]:
        try:
            combined = combined | annotation
        except TypeError:
            return Any
    return combined


def _is_unknown_annotation(annotation: Any) -> bool:
    return annotation in {Any, object}


def _raise_validation_error(
    validation_error_type: type[Exception] | None,
    selector: Selector,
    step_index: int,
    reason: str,
    *,
    error_prefix: str | None,
    root_label: str,
) -> NoReturn:
    if validation_error_type is None:
        raise _SelectorValidationFallback
    path = selector.render_path(root_label, upto=step_index + 1)
    raise validation_error_type(_prefixed_message(error_prefix, f"cannot resolve {path}: {reason}"))


def _raise_validation_index_error(
    validation_error_type: type[Exception] | None,
    selector: Selector,
    step_index: int,
    reason: str,
    *,
    error_prefix: str | None,
    root_label: str,
) -> NoReturn:
    if validation_error_type is None:
        raise _SelectorValidationFallback
    path = selector.render_path(root_label, upto=step_index + 1)
    raise validation_error_type(_prefixed_message(error_prefix, f"is out of bounds at {path}: {reason}"))


def _attribute_override(annotation: Any, attribute: str) -> Any:
    owner = annotation if isinstance(annotation, type) else get_origin(annotation)
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION

    module = getattr(owner, "__module__", "")
    name = getattr(owner, "__name__", "")
    if attribute == "shape" and (
        (module.startswith("numpy") and name == "ndarray")
        or (module.startswith("torch") and name == "Tensor")
    ):
        return tuple[int, ...]
    return _MISSING_ANNOTATION


def _is_typed_dict_annotation(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and issubclass(annotation, dict)
        and hasattr(annotation, "__annotations__")
        and hasattr(annotation, "__total__")
    )


def _mapping_annotation(annotation: Any) -> tuple[Any, Any] | None:
    if _is_typed_dict_annotation(annotation):
        return str, Any

    origin = get_origin(annotation)
    owner = origin if isinstance(origin, type) else annotation if isinstance(annotation, type) else None
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


def _annotation_accepts_value(annotation: Any, value: object) -> bool:
    if annotation in {Any, object}:
        return True
    if annotation in {None, _NONE_TYPE}:
        return value is None
    if _is_union_annotation(annotation):
        return any(_annotation_accepts_value(option, value) for option in get_args(annotation))
    origin = get_origin(annotation)
    if origin is not None:
        if repr(origin) == "typing.Literal":
            return value in get_args(annotation)
        return True
    if isinstance(annotation, type):
        try:
            return isinstance(value, annotation)
        except TypeError:
            return True
    return True


def _typed_dict_key_annotation(annotation: Any, key: str) -> Any:
    try:
        hints = get_type_hints(annotation)
    except Exception:
        hints = getattr(annotation, "__annotations__", {})
    return hints.get(key, _MISSING_ANNOTATION)


def _attribute_annotation(
    annotation: Any,
    attribute: str,
    *,
    validation_error_type: type[Exception] | None = None,
    owner_label: str | None = None,
) -> Any:
    if _is_unknown_annotation(annotation):
        return Any

    if annotation in {None, _NONE_TYPE}:
        if validation_error_type is not None:
            label = owner_label or "None"
            raise validation_error_type(f"{label} has no attribute {attribute!r}")
        return Any

    if _is_union_annotation(annotation):
        results: list[Any] = []
        errors: list[Exception] = []
        for option in get_args(annotation):
            try:
                results.append(
                    _attribute_annotation(
                        option,
                        attribute,
                        validation_error_type=validation_error_type,
                        owner_label=owner_label,
                    )
                )
            except Exception as exc:
                errors.append(exc)
        if errors and results:
            return Any
        if results:
            return _combine_annotations(*results)
        if errors and validation_error_type is not None:
            raise errors[0]
        return Any

    if _is_typed_dict_annotation(annotation):
        result = _typed_dict_key_annotation(annotation, attribute)
        if result is not _MISSING_ANNOTATION:
            return result
        if validation_error_type is not None:
            label = owner_label or _annotation_description(annotation)
            raise validation_error_type(f"{label} has no key {attribute!r}")
        return Any

    override = _attribute_override(annotation, attribute)
    if override is not _MISSING_ANNOTATION:
        return override

    owner = get_origin(annotation)
    if not isinstance(owner, type):
        owner = annotation if isinstance(annotation, type) else None
    if owner is None:
        return Any

    property_obj = getattr(owner, attribute, _MISSING_ANNOTATION)
    if isinstance(property_obj, property):
        try:
            if property_obj.fget is None:
                return Any
            return get_type_hints(property_obj.fget).get("return", Any)
        except Exception:
            return Any

    try:
        hints = get_type_hints(owner)
    except Exception:
        hints = getattr(owner, "__annotations__", {})
    if attribute in hints:
        return hints[attribute]
    if property_obj is not _MISSING_ANNOTATION:
        return Any
    if validation_error_type is not None:
        label = owner_label or _annotation_description(owner)
        raise validation_error_type(f"{label} has no attribute {attribute!r}")
    return Any


def _tuple_elements(annotation: Any) -> tuple[tuple[Any, ...], bool] | None:
    if isinstance(annotation, tuple):
        return annotation, False
    origin = get_origin(annotation)
    if origin is not tuple:
        return None
    args = get_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        return (args[0],), True
    return args, False


def _sequence_item_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {list, Sequence, MutableSequence}:
        args = get_args(annotation)
        return args[0] if args else Any
    owner = origin if isinstance(origin, type) else annotation if isinstance(annotation, type) else None
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION
    try:
        if issubclass(owner, MutableSequence):
            return Any
        if issubclass(owner, Sequence) and not issubclass(owner, (str, bytes, bytearray, Mapping)):
            return Any
    except TypeError:
        return _MISSING_ANNOTATION
    return _MISSING_ANNOTATION


def _mutable_sequence_item_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {list, MutableSequence}:
        args = get_args(annotation)
        return args[0] if args else Any
    owner = origin if isinstance(origin, type) else annotation if isinstance(annotation, type) else None
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION
    try:
        if issubclass(owner, MutableSequence):
            return Any
    except TypeError:
        return _MISSING_ANNOTATION
    return _MISSING_ANNOTATION


def _generic_indexable_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    owner = origin if isinstance(origin, type) else annotation if isinstance(annotation, type) else None
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION
    if issubclass(owner, (str, bytes, bytearray, Mapping)):
        return _MISSING_ANNOTATION
    if hasattr(owner, "__getitem__"):
        return Any
    return _MISSING_ANNOTATION


def _generic_writable_indexable_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    owner = origin if isinstance(origin, type) else annotation if isinstance(annotation, type) else None
    if not isinstance(owner, type):
        return _MISSING_ANNOTATION
    if issubclass(owner, (str, bytes, bytearray, Mapping)):
        return _MISSING_ANNOTATION
    if hasattr(owner, "__setitem__"):
        return Any
    return _MISSING_ANNOTATION

def _validate_read_step(
    annotation: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    if _is_unknown_annotation(annotation):
        return Any
    if annotation in {None, _NONE_TYPE}:
        _raise_validation_error(
            validation_error_type,
            selector,
            step_index,
            f"{_annotation_description(annotation)} does not support selector step {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    if _is_union_annotation(annotation):
        return _validate_read_union(
            annotation,
            part,
            selector,
            step_index,
            validation_error_type=validation_error_type,
            error_prefix=error_prefix,
            root_label=root_label,
        )
    match part:
        case str() as key if _is_typed_dict_annotation(annotation):
            return _validate_typed_dict_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case str() as key if _mapping_annotation(annotation) is not None:
            return _validate_mapping_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case str() as attribute:
            return _validate_attribute_step(
                annotation,
                attribute,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as key if _mapping_annotation(annotation) is not None:
            return _validate_mapping_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as index:
            return _validate_read_index_step(
                annotation,
                index,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
    raise AssertionError(f"Unsupported selector part: {part!r}")


def _validate_write_target(
    annotation: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    if _is_unknown_annotation(annotation):
        return Any
    if annotation in {None, _NONE_TYPE}:
        _raise_validation_error(
            validation_error_type,
            selector,
            step_index,
            f"{_annotation_description(annotation)} does not support writes",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    if _is_union_annotation(annotation):
        return _validate_write_union(
            annotation,
            part,
            selector,
            step_index,
            validation_error_type=validation_error_type,
            error_prefix=error_prefix,
            root_label=root_label,
        )
    match part:
        case str() as key if _is_typed_dict_annotation(annotation):
            return _validate_typed_dict_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case str() as key if _mapping_annotation(annotation) is not None:
            return _validate_mapping_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case str() as attribute:
            return _validate_attribute_write_target(
                annotation,
                attribute,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as key if _mapping_annotation(annotation) is not None:
            return _validate_mapping_key_step(
                annotation,
                key,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
        case int() as index:
            return _validate_write_index_target(
                annotation,
                index,
                selector,
                step_index,
                validation_error_type=validation_error_type,
                error_prefix=error_prefix,
                root_label=root_label,
            )
    raise AssertionError(f"Unsupported selector part: {part!r}")


def _validate_read_union(
    annotation: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    results: list[Any] = []
    errors: list[Exception] = []
    for option in get_args(annotation):
        try:
            results.append(
                _validate_read_step(
                    option,
                    part,
                    selector,
                    step_index,
                    validation_error_type=validation_error_type,
                    error_prefix=error_prefix,
                    root_label=root_label,
                )
            )
        except Exception as exc:
            errors.append(exc)

    if errors and results:
        return Any
    if results:
        return _combine_annotations(*results)
    if errors and validation_error_type is not None:
        raise errors[0]
    return Any


def _validate_write_union(
    annotation: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    results: list[Any] = []
    errors: list[Exception] = []
    for option in get_args(annotation):
        try:
            results.append(
                _validate_write_target(
                    option,
                    part,
                    selector,
                    step_index,
                    validation_error_type=validation_error_type,
                    error_prefix=error_prefix,
                    root_label=root_label,
                )
            )
        except Exception as exc:
            errors.append(exc)

    if errors and results:
        return Any
    if results:
        return _combine_annotations(*results)
    if errors and validation_error_type is not None:
        raise errors[0]
    return Any


def _validate_typed_dict_key_step(
    annotation: Any,
    part: str,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    result = _typed_dict_key_annotation(annotation, part)
    if result is not _MISSING_ANNOTATION:
        return result
    _raise_validation_error(
        validation_error_type,
        selector,
        step_index,
        f"{_annotation_description(annotation)} at {current_path} has no key {part!r}",
        error_prefix=error_prefix,
        root_label=root_label,
    )


def _validate_mapping_key_step(
    annotation: Any,
    part: SelectorPart,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    mapping_types = _mapping_annotation(annotation)
    if mapping_types is None:
        raise AssertionError("mapping key validation requires a mapping annotation")
    key_type, value_type = mapping_types
    if not _annotation_accepts_value(key_type, part):
        _raise_validation_error(
            validation_error_type,
            selector,
            step_index,
            f"{_annotation_description(annotation)} at {current_path} expects a key compatible with "
            f"{_annotation_description(key_type)}, got {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    return value_type


def _validate_attribute_step(
    annotation: Any,
    part: str,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    current_path = selector.render_path(root_label, upto=step_index)
    owner_label = f"{_annotation_description(annotation)} at {current_path}"
    return _attribute_annotation(
        annotation,
        part,
        validation_error_type=validation_error_type,
        owner_label=owner_label,
    )


def _validate_attribute_write_target(
    annotation: Any,
    part: str,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    attribute_annotation = _validate_attribute_step(
        annotation,
        part,
        selector,
        step_index,
        validation_error_type=validation_error_type,
        error_prefix=error_prefix,
        root_label=root_label,
    )

    owner = get_origin(annotation)
    if not isinstance(owner, type):
        owner = annotation if isinstance(annotation, type) else None
    if not isinstance(owner, type):
        return attribute_annotation

    descriptor = getattr(owner, part, _MISSING_ANNOTATION)
    if isinstance(descriptor, property) and descriptor.fset is None:
        current_path = selector.render_path(root_label, upto=step_index)
        _raise_validation_error(
            validation_error_type,
            selector,
            step_index,
            f"{_annotation_description(annotation)} at {current_path} has no writable attribute {part!r}",
            error_prefix=error_prefix,
            root_label=root_label,
        )
    return attribute_annotation


def _validate_read_index_step(
    annotation: Any,
    part: int,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    tuple_types = _tuple_elements(annotation)
    if tuple_types is not None:
        values, variadic = tuple_types
        if variadic:
            return values[0]
        if not (-len(values) <= part < len(values)):
            _raise_validation_index_error(
                validation_error_type,
                selector,
                step_index,
                f"{_annotation_description(annotation)} has length {len(values)}",
                error_prefix=error_prefix,
                root_label=root_label,
            )
        return values[part]

    sequence_item = _sequence_item_annotation(annotation)
    if sequence_item is not _MISSING_ANNOTATION:
        return sequence_item

    generic_indexable = _generic_indexable_annotation(annotation)
    if generic_indexable is not _MISSING_ANNOTATION:
        return generic_indexable

    current_path = selector.render_path(root_label, upto=step_index)
    _raise_validation_error(
        validation_error_type,
        selector,
        step_index,
        f"{_annotation_description(annotation)} at {current_path} is not indexable",
        error_prefix=error_prefix,
        root_label=root_label,
    )


def _validate_write_index_target(
    annotation: Any,
    part: int,
    selector: Selector,
    step_index: int,
    *,
    validation_error_type: type[Exception] | None,
    error_prefix: str | None,
    root_label: str,
) -> Any:
    tuple_types = _tuple_elements(annotation)
    if tuple_types is not None:
        current_path = selector.render_path(root_label, upto=step_index)
        _raise_validation_error(
            validation_error_type,
            selector,
            step_index,
            f"{_annotation_description(annotation)} at {current_path} is immutable",
            error_prefix=error_prefix,
            root_label=root_label,
        )

    sequence_item = _mutable_sequence_item_annotation(annotation)
    if sequence_item is not _MISSING_ANNOTATION:
        return sequence_item

    generic_indexable = _generic_writable_indexable_annotation(annotation)
    if generic_indexable is not _MISSING_ANNOTATION:
        return generic_indexable

    current_path = selector.render_path(root_label, upto=step_index)
    _raise_validation_error(
        validation_error_type,
        selector,
        step_index,
        f"{_annotation_description(annotation)} at {current_path} does not support item assignment",
        error_prefix=error_prefix,
        root_label=root_label,
    )
