from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pytest

from ml_pipes import Operator, Pipeline, PipelineValidationError
from ml_pipes.selector import Selector

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class Message:
    msg: str


@dataclass
class Member:
    name: str


@dataclass
class Club:
    members: list[Member] = field(default_factory=list)


@dataclass
class ClubCarrier:
    club: Club


@dataclass
class DataRecord:
    payload: Message | None = None
    text: str | None = None
    items: list[int] = field(default_factory=list)
    coords: tuple[int, int] = (0, 0)
    array: np.ndarray | None = None


class PlainRecord:
    def __init__(
        self,
        *,
        payload: Message | None = None,
        text: str | None = None,
        items: list[int] | None = None,
        coords: tuple[int, int] = (0, 0),
        array: np.ndarray | None = None,
    ) -> None:
        self.payload = payload
        self.text = text
        self.items = [0, 0] if items is None else items
        self.coords = coords
        self.array = array


class SlottedRecord:
    __slots__ = ("payload", "text", "items", "coords", "array")

    payload: Message | None
    text: str | None
    items: list[int]
    coords: tuple[int, int]
    array: np.ndarray | None

    def __init__(
        self,
        *,
        payload: Message | None = None,
        text: str | None = None,
        items: list[int] | None = None,
        coords: tuple[int, int] = (0, 0),
        array: np.ndarray | None = None,
    ) -> None:
        self.payload = payload
        self.text = text
        self.items = [0, 0] if items is None else items
        self.coords = coords
        self.array = array


class PayloadDict(TypedDict):
    msg: str


@dataclass
class ValidationCarrier:
    payload: Message
    text: str | None = None


class SlottedValidationCarrier:
    __slots__ = ("payload", "text")

    payload: Message
    text: str | None

    def __init__(self, payload: Message, text: str | None = None) -> None:
        self.payload = payload
        self.text = text


@dataclass
class TypedDictCarrier:
    payload: PayloadDict


@dataclass
class OptionalTypedDictCarrier:
    payload: PayloadDict | None = None
    text: str | None = None


class AcceptInt:
    def __call__(self, value: int) -> str:
        return str(value)


class AcceptStr:
    def __call__(self, value: str) -> int:
        return len(value)


class AcceptCarrier:
    def __call__(self, value: ValidationCarrier) -> str:
        return value.text or ""


@Operator
class SelectorProbe:
    def __init__(
        self,
        *,
        select_src: Any | None = None,
        select_target: Any | None = None,
    ) -> None:
        self.select_src = None if select_src is None else Selector.from_input(select_src)
        self.select_target = None if select_target is None else Selector.from_input(select_target)

    def __call__(self, current: Any) -> Any:
        selected = current
        if self.select_src is not None:
            selected = self.select_src.select_value(current, error_prefix=type(self).__name__)
        if self.select_target is None:
            return selected
        target = self.select_target.select_field(
            current,
            create_missing_mappings=True,
            error_prefix=type(self).__name__,
        )
        target.set(selected)
        return current

    def resolve_contract(
        self,
        current_output: Any,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del stored_annotations, expand_output_annotation
        selected = current_output
        if self.select_src is not None:
            selected = self.select_src.validate_read(
                current_output,
                validation_error_type=validation_error_type,
                error_prefix=f"{type(self).__name__}(src={self.select_src.steps!r})",
            )
        if self.select_target is not None:
            self.select_target.validate_write(
                current_output,
                validation_error_type=validation_error_type,
                error_prefix=f"{type(self).__name__}(target={self.select_target.steps!r})",
            )
            return (current_output,), current_output
        return (current_output,), selected


def test_selector_normalizes_dotted_string() -> None:
    selector = Selector.from_input("club.members.0")

    assert selector.steps == ("club", "members", 0)
    assert repr(selector) == "('club', 'members', 0)"


def test_selector_normalizes_bracket_string() -> None:
    selector = Selector.from_input("club.members[0].name")

    assert selector.steps == ("club", "members", 0, "name")
    assert selector.render_path("x") == "x.club.members[0].name"


def test_selector_from_input_flattens_mixed_forms() -> None:
    selector = Selector.from_input(("club.members", 0, ("name",)))

    assert selector.steps == ("club", "members", 0, "name")
    assert repr(selector) == "('club', 'members', 0, 'name')"


def test_selector_renders_trace_for_success_and_failure() -> None:
    carrier = ClubCarrier(club=Club(members=[Member("ann")]))
    selector = Selector.from_input("club.members.y")

    assert selector.render_path("x") == "x.club.members.y"

    with pytest.raises(TypeError, match=r"x\.club\.members\.y"):
        selector.select_value(carrier)
    with pytest.raises(TypeError, match="requires an integer index"):
        selector.select_value(carrier)


READ_CASES = [
    pytest.param(lambda: 5, None, lambda value: value == 5, None, id="identity_primitive"),
    pytest.param(lambda: PlainRecord(text="hello"), "text", lambda value: value == "hello", None, id="plain_object_attr"),
    pytest.param(lambda: DataRecord(payload=Message("hello")), "payload.msg", lambda value: value == "hello", None, id="dataclass_nested_attr"),
    pytest.param(lambda: SlottedRecord(payload=Message("hello")), "payload.msg", lambda value: value == "hello", None, id="slotted_nested_attr"),
    pytest.param(lambda: (10, 20), 1, lambda value: value == 20, None, id="tuple_index"),
    pytest.param(lambda: {"payload": {"msg": "hello"}}, "payload.msg", lambda value: value == "hello", None, id="dict_nested_key"),
    pytest.param(lambda: {1: "hello"}, 1, lambda value: value == "hello", None, id="dict_int_key"),
    pytest.param(lambda: [10, 20], 1, lambda value: value == 20, None, id="list_index"),
    pytest.param(lambda: DataRecord(coords=(10, 20)), ("coords", 1), lambda value: value == 20, None, id="tuple_form_selector"),
    pytest.param(
        lambda: ClubCarrier(club=Club(members=[Member("ann")])),
        "club.members[0].name",
        lambda value: value == "ann",
        None,
        id="bracket_path",
    ),
    pytest.param(
        lambda: np.zeros((3, 4), dtype=np.float32),
        "shape.0",
        lambda value: value == 3,
        None,
        id="ndarray_shape_then_index",
    ),
    pytest.param(
        lambda: np.array([10, 20], dtype=np.int64),
        1,
        lambda value: int(value) == 20,
        None,
        id="ndarray_direct_index",
    ),
]

if torch is not None:
    READ_CASES.extend(
        [
            pytest.param(
                lambda: torch.zeros((3, 4), dtype=torch.float32),
                "shape.0",
                lambda value: value == 3,
                None,
                id="torch_shape_then_index",
            ),
            pytest.param(
                lambda: torch.tensor([10, 20], dtype=torch.int64),
                1,
                lambda value: int(value.item()) == 20,
                None,
                id="torch_direct_index",
            ),
        ]
    )


@pytest.mark.parametrize("value_factory,selector_input,assert_value,error_match", READ_CASES)
def test_selector_runtime_read_cases(
    value_factory: Any,
    selector_input: Any,
    assert_value: Any,
    error_match: str | None,
) -> None:
    selector = Selector.from_input(selector_input)

    if error_match is not None:
        with pytest.raises(TypeError, match=error_match):
            selector.select_value(value_factory())
        return

    assert assert_value(selector.select_value(value_factory()))


WRITE_CASES = [
    pytest.param(lambda: {}, "msg", "hello", "msg", lambda value: value == "hello", None, id="dict_key"),
    pytest.param(lambda: {}, "payload.msg", "hello", "payload.msg", lambda value: value == "hello", None, id="dict_nested_auto_create"),
    pytest.param(lambda: {}, ("payload", "msg"), "hello", ("payload", "msg"), lambda value: value == "hello", None, id="tuple_form_target"),
    pytest.param(lambda: PlainRecord(), "text", "hello", "text", lambda value: value == "hello", None, id="plain_object_attr"),
    pytest.param(lambda: DataRecord(), "text", "hello", "text", lambda value: value == "hello", None, id="dataclass_attr"),
    pytest.param(lambda: SlottedRecord(), "text", "hello", "text", lambda value: value == "hello", None, id="slotted_attr"),
    pytest.param(lambda: [0, 0], 1, 7, 1, lambda value: value == 7, None, id="list_index"),
    pytest.param(lambda: DataRecord(items=[0, 0]), "items.1", 7, "items.1", lambda value: value == 7, None, id="object_list_index"),
    pytest.param(
        lambda: {"payload": PlainRecord()},
        "payload.text",
        "hello",
        "payload.text",
        lambda value: value == "hello",
        None,
        id="mapping_then_attr",
    ),
    pytest.param(
        lambda: np.zeros((2,), dtype=np.int64),
        1,
        7,
        1,
        lambda value: int(value) == 7,
        None,
        id="ndarray_index",
    ),
    pytest.param(lambda: (0, 0), 1, 7, None, None, "item assignment", id="tuple_index_parent_not_writable"),
    pytest.param(
        lambda: PlainRecord(),
        "missing.msg",
        "hello",
        None,
        None,
        "has no attribute",
        id="missing_object_parent_path",
    ),
]

if torch is not None:
    WRITE_CASES.append(
        pytest.param(
            lambda: torch.zeros((2,), dtype=torch.int64),
            1,
            7,
            1,
            lambda value: int(value.item()) == 7,
            None,
            id="torch_index",
        )
    )


@pytest.mark.parametrize(
    "value_factory,selector_input,item,readback_selector,assert_value,error_match",
    WRITE_CASES,
)
def test_selector_runtime_write_cases(
    value_factory: Any,
    selector_input: Any,
    item: object,
    readback_selector: Any | None,
    assert_value: Any,
    error_match: str | None,
) -> None:
    current = value_factory()
    selector = Selector.from_input(selector_input)

    if error_match is not None:
        with pytest.raises((TypeError, IndexError), match=error_match):
            selector.select_field(current, create_missing_mappings=True).set(item)
        return

    field = selector.select_field(current, create_missing_mappings=True)
    assert field.field_path == selector.render_path("x")
    assert field.parent_path == selector.render_path("x", upto=len(selector.steps) - 1)
    field.set(item)
    assert readback_selector is not None
    assert assert_value(Selector.from_input(readback_selector).select_value(current))


READ_VALIDATION_CASES = [
    pytest.param(ValidationCarrier, "payload.msg", str, None, id="dataclass_attr_path"),
    pytest.param(SlottedValidationCarrier, "payload.msg", str, None, id="slotted_attr_path"),
    pytest.param(tuple[str, int], 1, int, None, id="tuple_index_path"),
    pytest.param(list[int], 0, int, None, id="list_index_path"),
    pytest.param(dict[int, str], 1, str, None, id="dict_int_key_path"),
    pytest.param(TypedDictCarrier, "payload.msg", str, None, id="typed_dict_path"),
    pytest.param(OptionalTypedDictCarrier, "payload.msg", Any, None, id="optional_typed_dict_is_permissive"),
    pytest.param(dict[str, object], "payload.msg", Any, None, id="generic_dict_path_is_permissive"),
    pytest.param(np.ndarray, "shape.0", int, None, id="ndarray_shape_path"),
    pytest.param(np.ndarray, 0, Any, None, id="ndarray_direct_index_is_generic"),
    pytest.param(ValidationCarrier, "payload.missing", None, "has no attribute", id="missing_attr_fails"),
    pytest.param(tuple[str, int], 2, None, "out of bounds", id="tuple_index_out_of_bounds"),
]


@pytest.mark.parametrize("annotation,selector_input,expected,error_match", READ_VALIDATION_CASES)
def test_selector_validate_read_cases(
    annotation: Any,
    selector_input: Any,
    expected: Any,
    error_match: str | None,
) -> None:
    selector = Selector.from_input(selector_input)

    if error_match is not None:
        with pytest.raises(PipelineValidationError, match=error_match):
            selector.validate_read(annotation, validation_error_type=PipelineValidationError)
        return

    assert selector.validate_read(annotation, validation_error_type=PipelineValidationError) == expected


WRITE_VALIDATION_CASES = [
    pytest.param(ValidationCarrier, "text", str | None, None, id="existing_attr_target"),
    pytest.param(list[int], 1, int, None, id="list_target"),
    pytest.param(dict[int, str], 1, str, None, id="dict_int_key_target"),
    pytest.param(dict[str, object], "payload.msg", Any, None, id="generic_dict_target_is_permissive"),
    pytest.param(TypedDictCarrier, "payload.msg", str, None, id="typed_dict_target"),
    pytest.param(np.ndarray, 1, Any, None, id="ndarray_target"),
    pytest.param(ValidationCarrier, "missing", None, "has no attribute", id="missing_attr_target_fails"),
    pytest.param(tuple[str, int], 1, None, "immutable", id="tuple_target_is_not_writable"),
]


@pytest.mark.parametrize("annotation,selector_input,expected,error_match", WRITE_VALIDATION_CASES)
def test_selector_validate_write_cases(
    annotation: Any,
    selector_input: Any,
    expected: Any,
    error_match: str | None,
) -> None:
    selector = Selector.from_input(selector_input)

    if error_match is not None:
        with pytest.raises(PipelineValidationError, match=error_match):
            selector.validate_write(annotation, validation_error_type=PipelineValidationError)
        return

    assert selector.validate_write(annotation, validation_error_type=PipelineValidationError) == expected


def _validate_selector_probe(
    *,
    pipeline_input_type: Any,
    select_src: Any | None = None,
    select_target: Any | None = None,
    downstream: Any | None = None,
    strict: bool = True,
) -> object:
    operators = [SelectorProbe(select_src=select_src, select_target=select_target)]
    if downstream is not None:
        operators.append(downstream)
    return Pipeline(operators).validate(
        pipeline_input_type=pipeline_input_type,
        strict=strict,
    )


SRC_VALIDATION_CASES = [
    pytest.param(ValidationCarrier, "payload.msg", AcceptStr(), True, None, ValidationCarrier, int, id="src_only_attr_path"),
    pytest.param(tuple[str, int], 1, AcceptInt(), True, None, tuple[str, int], str, id="src_only_tuple_index"),
    pytest.param(OptionalTypedDictCarrier, "payload.msg", AcceptInt(), False, None, OptionalTypedDictCarrier, str, id="src_only_optional_path_is_permissive"),
    pytest.param(ValidationCarrier, "payload.missing", None, True, "has no attribute", None, None, id="src_only_missing_attr_fails"),
]


@pytest.mark.parametrize(
    "pipeline_input_type,select_src,downstream,strict,error_match,expected_input,expected_output",
    SRC_VALIDATION_CASES,
)
def test_selector_probe_validation_src_only_cases(
    pipeline_input_type: Any,
    select_src: Any,
    downstream: Any | None,
    strict: bool,
    error_match: str | None,
    expected_input: Any,
    expected_output: Any,
) -> None:
    if error_match is not None:
        with pytest.raises(PipelineValidationError, match=error_match):
            _validate_selector_probe(
                pipeline_input_type=pipeline_input_type,
                select_src=select_src,
                downstream=downstream,
                strict=strict,
            )
        return

    contract = _validate_selector_probe(
        pipeline_input_type=pipeline_input_type,
        select_src=select_src,
        downstream=downstream,
        strict=strict,
    )
    assert contract.input_type == expected_input
    assert contract.output_type == expected_output


TARGET_VALIDATION_CASES = [
    pytest.param(ValidationCarrier, None, "text", AcceptCarrier(), None, ValidationCarrier, str, id="target_only_attr"),
    pytest.param(dict[str, object], None, "payload.msg", None, None, dict[str, object], dict[str, object], id="target_only_generic_dict"),
    pytest.param(tuple[str, int], None, 1, None, "immutable", None, None, id="target_only_tuple_fails"),
    pytest.param(ValidationCarrier, None, "missing", None, "has no attribute", None, None, id="target_only_missing_attr_fails"),
]


@pytest.mark.parametrize(
    "pipeline_input_type,select_src,select_target,downstream,error_match,expected_input,expected_output",
    TARGET_VALIDATION_CASES,
)
def test_selector_probe_validation_target_only_cases(
    pipeline_input_type: Any,
    select_src: Any | None,
    select_target: Any,
    downstream: Any | None,
    error_match: str | None,
    expected_input: Any,
    expected_output: Any,
) -> None:
    if error_match is not None:
        with pytest.raises(PipelineValidationError, match=error_match):
            _validate_selector_probe(
                pipeline_input_type=pipeline_input_type,
                select_src=select_src,
                select_target=select_target,
                downstream=downstream,
            )
        return

    contract = _validate_selector_probe(
        pipeline_input_type=pipeline_input_type,
        select_src=select_src,
        select_target=select_target,
        downstream=downstream,
    )
    assert contract.input_type == expected_input
    assert contract.output_type == expected_output


SRC_TARGET_VALIDATION_CASES = [
    pytest.param(ValidationCarrier, "payload.msg", "text", AcceptCarrier(), None, ValidationCarrier, str, id="src_target_attr_to_attr"),
    pytest.param(dict[str, object], "payload.msg", "cleaned", None, None, dict[str, object], dict[str, object], id="src_target_generic_dict"),
    pytest.param(ValidationCarrier, "payload.missing", "text", None, "has no attribute", None, None, id="src_target_missing_src_fails"),
    pytest.param(ValidationCarrier, "payload.msg", "missing", None, "has no attribute", None, None, id="src_target_missing_target_fails"),
    pytest.param(tuple[str, int], 0, 1, None, "immutable", None, None, id="src_target_tuple_target_fails"),
]


@pytest.mark.parametrize(
    "pipeline_input_type,select_src,select_target,downstream,error_match,expected_input,expected_output",
    SRC_TARGET_VALIDATION_CASES,
)
def test_selector_probe_validation_src_and_target_cases(
    pipeline_input_type: Any,
    select_src: Any,
    select_target: Any,
    downstream: Any | None,
    error_match: str | None,
    expected_input: Any,
    expected_output: Any,
) -> None:
    if error_match is not None:
        with pytest.raises(PipelineValidationError, match=error_match):
            _validate_selector_probe(
                pipeline_input_type=pipeline_input_type,
                select_src=select_src,
                select_target=select_target,
                downstream=downstream,
            )
        return

    contract = _validate_selector_probe(
        pipeline_input_type=pipeline_input_type,
        select_src=select_src,
        select_target=select_target,
        downstream=downstream,
    )
    assert contract.input_type == expected_input
    assert contract.output_type == expected_output


def test_selector_probe_runtime_src_and_target_case() -> None:
    carrier = ValidationCarrier(payload=Message("hello"))

    result = SelectorProbe(select_src="payload.msg", select_target="text")(carrier)

    assert result is carrier
    assert carrier.text == "hello"
