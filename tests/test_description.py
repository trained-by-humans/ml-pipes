from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml_pipes import (
    AsType,
    Batch,
    Extract,
    FilterPredictionsByClass,
    FilterPredictionsByScore,
    FilterTensorsByScore,
    Gather,
    InvocationTrace,
    Operator,
    OperatorArgument,
    OperatorDescription,
    Pipeline,
    Recall,
    Scatter,
    Store,
    TraceCollector,
    TracingConfig,
    UnBatch,
    embed,
)
from ml_pipes.operator import get_operator_constructor_signature


@Operator
class ConfiguredIntToString:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def __call__(self, value: int) -> str:
        return f"{self.prefix}{value}"


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


class FloatToBool:
    def __call__(self, value: float) -> bool:
        return value > 0


class ListIdentity:
    def __call__(self, values: list[int]) -> list[int]:
        return values


class LegacyConfiguredIntToString:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def __call__(self, value: int) -> str:
        return f"{self.prefix}{value}"


class ReprOnlyLegacyOp:
    def __call__(self, value: int) -> int:
        return value

    def __repr__(self) -> str:
        return "LegacyCustom"


class _Capture(TraceCollector):
    def on_trace(self, trace: InvocationTrace) -> None:
        del trace


def _named_identity(value: Any) -> Any:
    return value


_MODULE_LAMBDA = lambda value: value


class _CallableArg:
    def __call__(self, value: Any) -> Any:
        return value


class _CallableArgWithRepr:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, value: Any) -> Any:
        return value

    def __repr__(self) -> str:
        return f"_CallableArgWithRepr(name={self.name!r})"


class _CustomArg:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"_CustomArg(name={self.name!r})"


@Operator
class PrimitiveArgsOp:
    def __init__(
        self,
        flag: bool,
        count: int,
        ratio: float,
        label: str,
        payload: bytes,
        maybe: None,
    ) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class StructuredArgsOp:
    def __init__(
        self,
        coords: tuple[int, int],
        names: list[str],
        mapping: dict[str, int],
        classes: set[int],
        frozen: frozenset[str],
    ) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class CallableArgOp:
    def __init__(self, fn: Any) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class CustomObjectArgOp:
    def __init__(self, payload: Any) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class VariadicArgsOp:
    def __init__(self, *items: str, **named: Any) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class PipelineParamOp:
    def __init__(self, pipeline: str, other: str) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value


@Operator
class CustomRenderedOp:
    def __init__(self, value: str, flag: bool = False) -> None:
        self.value = value
        self.flag = flag

    def __call__(self, value: int) -> int:
        return value

    def __repr__(self) -> str:
        return f"Custom<{self.value!r}>"

    def describe(self, *, show_defaults: bool = False) -> str:
        suffix = ", flag=False" if show_defaults else ""
        return f"CustomDescribe(value={self.value!r}{suffix})"


@Operator
class DescribeOnlyOp:
    def __init__(self, value: str, flag: bool = False) -> None:
        self.value = value
        self.flag = flag

    def __call__(self, value: int) -> int:
        return value

    def describe(self, *, show_defaults: bool = False) -> str:
        suffix = ", flag=False" if show_defaults else ""
        return f"DescribeOnly(value={self.value!r}{suffix})"


@Operator
class InvalidRenderedOp:
    def __init__(self, value: str, flag: bool = False) -> None:
        pass

    def __call__(self, value: int) -> int:
        return value

    def __repr__(self) -> str:
        raise RuntimeError("bad repr")

    def describe(self, *, show_defaults: bool = False) -> int:
        del show_defaults
        return 123


def _pipeline_text(*operators: str) -> str:
    if not operators:
        return "Pipeline([])"
    return "Pipeline([\n" + "\n".join(f"  {operator}," for operator in operators) + "\n])"


def _argument_dict(operator: Any, *, include_defaults: bool = False) -> dict[str, Any]:
    arguments = OperatorDescription.from_operator(operator).arguments
    if not include_defaults:
        arguments = tuple(argument for argument in arguments if argument.is_passed)
    return OperatorArgument.to_dict(arguments)


def test_pipeline_repr_renders_flat_operator_chain_with_named_args():
    pipeline = Pipeline([ConfiguredIntToString(prefix="id:"), StringToFloat()])
    description = pipeline.describe()

    assert isinstance(description.operators[0], OperatorDescription)
    assert [(operator.name, operator.passed_args) for operator in description.operators] == [
        ("ConfiguredIntToString", {"prefix": "id:"}),
        ("StringToFloat", {}),
    ]
    assert repr(pipeline) == _pipeline_text(
        "ConfiguredIntToString(prefix='id:')",
        "StringToFloat()",
    )
    assert repr(description) == repr(pipeline)


def test_operator_argument_to_dict_can_expand_defaults_on_read():
    operator = ConfiguredIntToString()

    assert _argument_dict(operator) == {}
    assert _argument_dict(operator, include_defaults=True) == {"prefix": ""}


def test_decorated_operator_describe_returns_operator_description():
    description = ConfiguredIntToString().describe()

    assert isinstance(description, OperatorDescription)
    assert description.name == "ConfiguredIntToString"
    assert description.default_args == {"prefix": ""}
    assert description.constructor_signature is not None
    assert tuple(description.constructor_signature.parameters) == ("prefix",)


def test_operator_rejects_slotted_classes():
    with pytest.raises(TypeError, match="captured constructor args"):
        @Operator
        class SlottedOp:
            __slots__ = ("prefix",)

            def __init__(self, prefix: str = "") -> None:
                self.prefix = prefix

            def __call__(self, value: int) -> int:
                return value


def test_operator_rejects_slotted_dataclass_classes():
    with pytest.raises(TypeError, match="captured constructor args"):
        @Operator
        @dataclass(slots=True)
        class SlottedDataclassOp:
            prefix: str = ""

            def __call__(self, value: int) -> int:
                return value


def test_function_operator_has_no_constructor_signature():
    assert get_operator_constructor_signature(_named_identity) is None


def test_pipeline_repr_uses_function_name_for_function_operators():
    assert repr(Pipeline([_named_identity])) == _pipeline_text("_named_identity")


def test_describe_formats_primitive_constructor_args():
    description = Pipeline([
        PrimitiveArgsOp(
            flag=True,
            count=3,
            ratio=0.5,
            label="hello",
            payload=b"data",
            maybe=None,
        )
    ]).describe()

    assert description.operators[0].passed_args == {
        "flag": True,
        "count": 3,
        "ratio": 0.5,
        "label": "hello",
        "payload": b"data",
        "maybe": None,
    }
    assert repr(description) == _pipeline_text(
        "PrimitiveArgsOp(flag=True, count=3, ratio=0.5, label='hello', payload=b'data', maybe=None)"
    )


def test_describe_formats_structured_constructor_args():
    description = Pipeline([
        StructuredArgsOp(
            coords=(1, 2),
            names=["a", "b"],
            mapping={"left": 1, "right": 2},
            classes={2, 0},
            frozen=frozenset({"beta", "alpha"}),
        )
    ]).describe()

    assert description.operators[0].passed_args == {
        "coords": (1, 2),
        "names": ["a", "b"],
        "mapping": {"left": 1, "right": 2},
        "classes": {0, 2},
        "frozen": frozenset({"alpha", "beta"}),
    }
    assert repr(description) == _pipeline_text(
        "StructuredArgsOp(coords=(1, 2), names=['a', 'b'], mapping={'left': 1, 'right': 2}, "
        "classes={0, 2}, frozen=frozenset({'alpha', 'beta'}))"
    )


def test_operator_args_capture_lambda_and_render_with_lambda_label():
    operator = CallableArgOp(_MODULE_LAMBDA)
    description = Pipeline([operator]).describe()

    assert _argument_dict(operator)["fn"] is _MODULE_LAMBDA
    assert repr(description) == _pipeline_text("CallableArgOp(<lambda>)")


def test_operator_args_capture_function_and_render_with_function_name():
    operator = CallableArgOp(_named_identity)
    description = Pipeline([operator]).describe()

    assert _argument_dict(operator)["fn"] is _named_identity
    assert repr(description) == _pipeline_text("CallableArgOp(_named_identity)")


def test_operator_args_capture_callable_object_and_render_with_type_name():
    callable_object = _CallableArg()
    operator = CallableArgOp(callable_object)
    description = Pipeline([operator]).describe()

    assert _argument_dict(operator)["fn"] is callable_object
    assert repr(description) == _pipeline_text("CallableArgOp(_CallableArg)")


def test_operator_args_capture_callable_object_with_custom_repr():
    callable_object = _CallableArgWithRepr("mapper")
    operator = CallableArgOp(callable_object)
    description = Pipeline([operator]).describe()

    assert _argument_dict(operator)["fn"] is callable_object
    assert repr(description) == _pipeline_text(
        "CallableArgOp(_CallableArgWithRepr(name='mapper'))"
    )


def test_operator_args_capture_custom_object_and_render_with_repr():
    payload = _CustomArg("example")
    operator = CustomObjectArgOp(payload)
    description = Pipeline([operator]).describe()

    assert _argument_dict(operator)["payload"] is payload
    assert repr(description) == _pipeline_text(
        "CustomObjectArgOp(_CustomArg(name='example'))"
    )


def test_operator_argument_to_dict_captures_varargs_and_kwargs():
    operator = VariadicArgsOp("first", "second", alpha=1, beta="two")

    assert _argument_dict(operator) == {
        "items": ("first", "second"),
        "named": {"alpha": 1, "beta": "two"},
    }
    assert repr(Pipeline([operator])) == _pipeline_text(
        "VariadicArgsOp('first', 'second', alpha=1, beta='two')"
    )


def test_operator_capture_keeps_real_pipeline_parameter_aligned():
    operator = PipelineParamOp("inner", "other")
    description = operator.describe()

    assert description.passed_args == {
        "pipeline": "inner",
        "other": "other",
    }
    assert tuple(description.constructor_signature.parameters) == ("pipeline", "other")
    assert repr(operator) == "PipelineParamOp('inner', 'other')"
    assert Pipeline([operator]).describe().operators[0].passed_args == {
        "pipeline": "inner",
        "other": "other",
    }


def test_describe_prints_requested_view_once_for_top_level_call(capsys):
    pipeline = Pipeline([ConfiguredIntToString()])
    description = pipeline.describe(show_defaults=True)

    captured = capsys.readouterr()

    assert captured.out == description.render(show_defaults=True) + "\n"
    assert repr(description) == _pipeline_text("ConfiguredIntToString()")
    assert description.render(show_defaults=True) == _pipeline_text(
        "ConfiguredIntToString(prefix='')"
    )


def test_operator_description_tracks_passed_and_default_args():
    description = Pipeline([ConfiguredIntToString()]).describe()
    operator = description.operators[0]

    assert operator.passed_args == {}
    assert operator.default_args == {"prefix": ""}
    assert operator.all_args == {"prefix": ""}
    assert repr(operator) == "ConfiguredIntToString()"
    assert operator.render(show_defaults=True, verbose=True) == "ConfiguredIntToString(prefix='')"


def test_describe_renders_context_operators_with_captured_args():
    description = Pipeline([Store("saved"), Recall("saved")]).describe()

    assert [operator.passed_args for operator in description.operators] == [
        {"name": "saved"},
        {"name": "saved"},
    ]
    assert repr(description) == _pipeline_text("Store('saved')", "Recall('saved')")


def test_describe_keeps_batch_region_operators_in_chain_order():
    description = Pipeline([Batch(size=2), ListIdentity(), UnBatch()]).describe()

    assert [operator.name for operator in description.operators] == [
        "Batch",
        "ListIdentity",
        "UnBatch",
    ]
    assert repr(description) == _pipeline_text("Batch(size=2)", "ListIdentity()", "UnBatch()")


def test_describe_keeps_scatter_region_operators_in_chain_order():
    description = Pipeline([Scatter(max_concurrency=1), ConfiguredIntToString(), Gather()]).describe()

    assert [operator.name for operator in description.operators] == [
        "Scatter",
        "ConfiguredIntToString",
        "Gather",
    ]
    assert repr(description) == _pipeline_text(
        "Scatter(max_concurrency=1)",
        "ConfiguredIntToString()",
        "Gather()",
    )


def test_describe_keeps_embed_flat_and_opaque():
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe()

    assert len(description.operators) == 2
    assert [operator.name for operator in description.operators] == ["Embed", "FloatToBool"]
    assert description.operators[0].passed_args["pipeline"] is inner
    assert repr(description) == _pipeline_text("Embed()", "FloatToBool()")


def test_describe_verbose_expands_embed_pipeline(capsys):
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    pipeline = Pipeline([embed(inner), FloatToBool()])
    description = pipeline.describe(verbose=True)

    captured = capsys.readouterr()

    assert description.render(verbose=True) == _pipeline_text(
        "Embed(Pipeline([\n    ConfiguredIntToString(),\n    StringToFloat(),\n  ]))",
        "FloatToBool()",
    )
    assert captured.out == description.render(verbose=True) + "\n"


def test_describe_ignores_custom_operator_labels_when_present():
    pipeline = Pipeline(
        [ConfiguredIntToString(), StringToFloat()],
        tracing=TracingConfig(collector=_Capture(), operator_labels=["to_str", "to_float"]),
    )

    description = pipeline.describe()

    assert [operator.name for operator in description.operators] == [
        "ConfiguredIntToString",
        "StringToFloat",
    ]


def test_describe_preserves_extract_constructor_config():
    description = Pipeline([Extract("output0", as_="preds")]).describe()

    assert description.operators[0].passed_args == {
        "names": ("output0",),
        "as_": "preds",
    }
    assert repr(description) == _pipeline_text("Extract('output0', as_='preds')")


def test_describe_preserves_explicit_none_constructor_args():
    description = Pipeline([Extract("output0", as_=None)]).describe()

    assert description.operators[0].passed_args == {
        "names": ("output0",),
        "as_": None,
    }
    assert repr(description) == _pipeline_text("Extract('output0', as_=None)")


def test_describe_captures_wrapper_constructor_args():
    assert Pipeline([FilterPredictionsByClass({0, 2})]).describe().operators[0].passed_args == {
        "classes": {0, 2},
    }
    assert repr(Pipeline([FilterPredictionsByClass({0, 2})])) == _pipeline_text(
        "FilterPredictionsByClass({0, 2})"
    )

    assert Pipeline([FilterPredictionsByScore(0.7)]).describe().operators[0].passed_args == {
        "min_score": 0.7,
    }
    assert repr(Pipeline([FilterPredictionsByScore(0.7)])) == _pipeline_text(
        "FilterPredictionsByScore(0.7)"
    )

    assert Pipeline([FilterTensorsByScore("boxes", score="scores", min_score=0.75)]).describe().operators[0].passed_args == {
        "srcs": ("boxes",),
        "score": "scores",
        "min_score": 0.75,
    }
    assert repr(Pipeline([FilterTensorsByScore("boxes", score="scores", min_score=0.75)])) == _pipeline_text(
        "FilterTensorsByScore('boxes', score='scores', min_score=0.75)"
    )

    assert Pipeline([AsType("float16")]).describe().operators[0].passed_args == {
        "dtype": "float16",
    }
    assert repr(Pipeline([AsType("float16")])) == _pipeline_text("AsType('float16')")


def test_describe_undecorated_custom_operator_shows_no_args():
    description = Pipeline([LegacyConfiguredIntToString(prefix="id:")]).describe()

    assert description.operators[0].passed_args == {}
    assert repr(description) == _pipeline_text("LegacyConfiguredIntToString()")


def test_description_ignores_custom_operator_repr():
    description = Pipeline([ReprOnlyLegacyOp()]).describe()

    assert repr(description) == _pipeline_text("ReprOnlyLegacyOp()")
    assert description.render(verbose=True) == _pipeline_text("ReprOnlyLegacyOp()")


def test_pipeline_ignores_custom_operator_repr_and_describe():
    pipeline = Pipeline([CustomRenderedOp("x")])
    description = pipeline.describe(show_defaults=True)

    assert repr(pipeline) == _pipeline_text("CustomRenderedOp('x')")
    assert description.render(show_defaults=True, verbose=True) == _pipeline_text(
        "CustomRenderedOp('x', flag=False)"
    )


def test_pipeline_ignores_invalid_custom_operator_repr_and_describe():
    pipeline = Pipeline([InvalidRenderedOp("x")])
    description = pipeline.describe(show_defaults=True)

    assert repr(pipeline) == _pipeline_text("InvalidRenderedOp('x')")
    assert description.render(show_defaults=True, verbose=True) == _pipeline_text(
        "InvalidRenderedOp('x', flag=False)"
    )


def test_default_operator_repr_falls_back_when_describe_returns_non_description():
    operator = DescribeOnlyOp("x")

    assert repr(operator) == "DescribeOnlyOp('x')"
    assert operator.describe() == "DescribeOnly(value='x')"
