import inspect

from ml_pipes._typing.inspection import resolve_unary_callable_annotations
from ml_pipes._typing.signatures import validate_unary_callable_signature


def test_unary_callable_helpers_share_variadic_positional_policy() -> None:
    def stringify(*values: int) -> str:
        return ",".join(str(value) for value in values)

    parameter = validate_unary_callable_signature(
        stringify,
        label="Map fn",
        source_label="the operator invokes it with the current value",
        error_type=TypeError,
    )
    annotations = resolve_unary_callable_annotations(stringify)

    assert parameter.kind is inspect.Parameter.VAR_POSITIONAL
    assert annotations.parameter_annotations == (int,)
    assert annotations.return_annotation is str
