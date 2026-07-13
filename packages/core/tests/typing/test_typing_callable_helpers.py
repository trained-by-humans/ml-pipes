import inspect
from unittest.mock import patch

import pytest

from ml_pipes._typing.inspection import (
    bind_method_call_parameter_names,
    resolve_callable_annotations,
    resolve_callable_signature_annotations,
)
from ml_pipes._typing.signatures import (
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
    annotations = resolve_callable_annotations(stringify)

    assert parameter.kind is inspect.Parameter.VAR_POSITIONAL
    assert annotations.parameter_annotations == (int,)
    assert annotations.return_annotation is str


def test_resolve_callable_annotations_orders_positional_annotations() -> None:
    def predicate(value: int, expected: str = "x") -> bool:
        return str(value) == expected

    annotations = resolve_callable_annotations(predicate)

    assert annotations.parameter_annotations == (int, str)
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
    assert tuple(parameter.annotation for parameter in annotations.parameters) == (
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


def test_bind_method_call_parameter_names_hides_inspect_binding_details() -> None:
    class Filterable:
        def filter(self, mask: list[bool], *, limit: int = 0) -> None:
            del mask, limit

    mask_token = object()
    limit_token = object()

    parameter_names = bind_method_call_parameter_names(
        Filterable.filter,
        (mask_token,),
        {"limit": limit_token},
    )

    assert parameter_names == {
        mask_token: "mask",
        limit_token: "limit",
    }


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
