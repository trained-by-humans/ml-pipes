import inspect
from unittest.mock import patch

import pytest

from ml_pipes._typing.signatures import (
    match_method_signatures,
    resolve_callable_signature_annotations,
    validate_callable_signature,
    validate_nullary_callable_signature,
    validate_unary_callable_signature,
)


def test_callable_signature_helpers_share_variadic_positional_policy() -> None:
    def stringify(*values: int) -> str:
        return ",".join(str(value) for value in values)

    parameter = validate_callable_signature(
        stringify,
        label="Map fn",
        argument_label="the current value",
        error_type=TypeError,
    )
    annotations = resolve_callable_signature_annotations(stringify)

    assert parameter.kind is inspect.Parameter.VAR_POSITIONAL
    assert annotations.parameter_annotations == (int,)
    assert annotations.return_annotation is str


def test_resolve_callable_signature_annotations_orders_parameters() -> None:
    def predicate(value: int, expected: str = "x") -> bool:
        return str(value) == expected

    annotations = resolve_callable_signature_annotations(predicate)

    assert annotations.parameter_annotations == (
        int,
        str,
    )
    assert annotations.return_annotation is bool


def test_resolve_callable_signature_annotations_preserves_keyword_only_parameters() -> None:
    def predicate(value: int, *, expected: str = "x") -> bool:
        return str(value) == expected

    annotations = resolve_callable_signature_annotations(predicate)

    assert annotations.is_inspectable is True
    assert tuple(parameter.parameter.name for parameter in annotations.parameters) == (
        "value",
        "expected",
    )
    assert tuple(parameter.parameter.kind for parameter in annotations.parameters) == (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    assert annotations.parameter_annotations == (
        int,
        str,
    )
    assert tuple(
        parameter.parameter.default is not inspect.Parameter.empty
        for parameter in annotations.parameters
    ) == (
        False,
        True,
    )
    assert annotations.return_annotation is bool


def test_resolve_callable_signature_annotations_preserves_instance_method_receiver() -> None:
    class Filterable:
        def filter(self, value: int) -> str:
            return str(value)

    annotations = resolve_callable_signature_annotations(Filterable.filter)

    assert tuple(parameter.parameter.name for parameter in annotations.parameters) == (
        "self",
        "value",
    )
    assert annotations.parameter_annotations == (None, int)
    assert annotations.return_annotation is str


def test_match_method_signatures_accepts_exact_callable_surface() -> None:
    class ProtocolLike:
        def filter(self, value: int, *, limit: int = 0) -> str:
            return str(value + limit)

    class Implementation:
        def filter(self, value: int, *, limit: int = 0) -> str:
            return str(value + limit)

    matched_signatures = match_method_signatures(
        Implementation,
        ProtocolLike,
        "filter",
    )

    assert matched_signatures is not None
    source_signature, target_signature = matched_signatures
    assert tuple(
        parameter.parameter.name for parameter in source_signature.parameters
    ) == ("value", "limit")
    assert tuple(
        parameter.parameter.name for parameter in target_signature.parameters
    ) == ("value", "limit")


def test_match_method_signatures_rejects_keyword_visible_parameter_name_mismatch() -> None:
    class ProtocolLike:
        def mix(self, left: int, right: int) -> int:
            return left + right

    class Implementation:
        def mix(self, right: int, left: int) -> int:
            return left + right

    assert match_method_signatures(
        Implementation,
        ProtocolLike,
        "mix",
    ) is None


def test_match_method_signatures_rejects_default_value_mismatch() -> None:
    class ProtocolLike:
        def filter(self, value: int, *, limit: int = 0) -> str:
            return str(value + limit)

    class Implementation:
        def filter(self, value: int, *, limit: int = 1) -> str:
            return str(value + limit)

    assert match_method_signatures(
        Implementation,
        ProtocolLike,
        "filter",
    ) is None


def test_match_method_signatures_rejects_missing_return_annotation() -> None:
    class ProtocolLike:
        def filter(self, value: int) -> str:
            return str(value)

    class Implementation:
        def filter(self, value: int):
            return str(value)

    assert match_method_signatures(
        Implementation,
        ProtocolLike,
        "filter",
    ) is None


def test_unary_callable_signature_reuses_public_callable_validation() -> None:
    def stringify(value: int) -> str:
        return str(value)

    parameter = validate_unary_callable_signature(
        stringify,
        label="Map fn",
        argument_label="the current value",
        error_type=TypeError,
    )

    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_nullary_and_positional_callable_validation_share_inspection_failure_path() -> None:
    with patch(
        "ml_pipes._typing.signatures.inspect.signature",
        side_effect=ValueError("signature unavailable"),
    ):
        with pytest.raises(
            TypeError,
            match="Map fn must expose an inspectable call signature because "
            "the current value is passed by position",
        ):
            validate_callable_signature(
                lambda value: value,
                label="Map fn",
                argument_label="the current value",
                error_type=TypeError,
            )

        with pytest.raises(
            TypeError,
            match="WrapMappingInObject state_factory must expose an inspectable call signature because "
            "it is invoked without arguments",
        ):
            validate_nullary_callable_signature(
                lambda: object(),
                label="WrapMappingInObject state_factory",
                error_type=TypeError,
            )
