from __future__ import annotations

from typing import Any

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
    Pipeline,
    Recall,
    Scatter,
    Store,
    TraceCollector,
    TracingConfig,
    UnBatch,
    embed,
)
from ml_pipes.operator import get_operator_args, get_operator_constructor_signature


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


class _Capture(TraceCollector):
    def on_trace(self, trace: InvocationTrace) -> None:
        del trace


def _named_identity(value: Any) -> Any:
    return value


_MODULE_LAMBDA = lambda value: value


class _CallableArg:
    def __call__(self, value: Any) -> Any:
        return value


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


def test_describe_renders_flat_operator_chain_with_named_args():
    description = Pipeline([ConfiguredIntToString(prefix="id:"), StringToFloat()]).describe()

    assert [(step.label, step.kind, step.operator_args) for step in description.steps] == [
        ("ConfiguredIntToString", "operator", {"prefix": "id:"}),
        ("StringToFloat", "operator", {}),
    ]
    assert repr(description) == (
        "Pipeline[\n"
        "  0:ConfiguredIntToString(prefix='id:')\n"
        "  1:StringToFloat()\n"
        "]"
    )

def test_get_operator_args_can_expand_defaults_on_read():
    operator = ConfiguredIntToString()

    assert get_operator_args(operator) == {}
    assert get_operator_args(operator, include_defaults=True) == {"prefix": ""}


def test_function_operator_has_no_constructor_signature():
    assert get_operator_constructor_signature(_named_identity) is None


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

    assert description.steps[0].operator_args == {
        "flag": True,
        "count": 3,
        "ratio": 0.5,
        "label": "hello",
        "payload": b"data",
        "maybe": None,
    }
    assert (
        repr(description)
        == "Pipeline[\n"
        "  0:PrimitiveArgsOp(flag=True, count=3, ratio=0.5, label='hello', payload=b'data', maybe=None)\n"
        "]"
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

    assert description.steps[0].operator_args == {
        "coords": (1, 2),
        "names": ["a", "b"],
        "mapping": {"left": 1, "right": 2},
        "classes": {0, 2},
        "frozen": frozenset({"alpha", "beta"}),
    }
    assert (
        repr(description)
        == "Pipeline[\n"
        "  0:StructuredArgsOp(coords=(1, 2), names=['a', 'b'], mapping={'left': 1, 'right': 2}, "
        "classes={0, 2}, frozen=frozenset({'alpha', 'beta'}))\n"
        "]"
    )


def test_operator_args_capture_lambda_and_describe_with_lambda_label():
    operator = CallableArgOp(_MODULE_LAMBDA)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is _MODULE_LAMBDA
    assert repr(description) == "Pipeline[\n  0:CallableArgOp(fn=<lambda>)\n]"


def test_operator_args_capture_function_and_describe_with_function_name():
    operator = CallableArgOp(_named_identity)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is _named_identity
    assert repr(description) == "Pipeline[\n  0:CallableArgOp(fn=_named_identity)\n]"


def test_operator_args_capture_callable_object_and_describe_with_type_name():
    callable_object = _CallableArg()
    operator = CallableArgOp(callable_object)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is callable_object
    assert repr(description) == "Pipeline[\n  0:CallableArgOp(fn=_CallableArg)\n]"


def test_operator_args_capture_custom_object_and_describe_with_repr():
    payload = _CustomArg("example")
    operator = CustomObjectArgOp(payload)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["payload"] is payload
    assert repr(description) == "Pipeline[\n  0:CustomObjectArgOp(payload=_CustomArg(name='example'))\n]"


def test_get_operator_args_captures_varargs_and_kwargs():
    operator = VariadicArgsOp("first", "second", alpha=1, beta="two")

    assert get_operator_args(operator) == {
        "items": ("first", "second"),
        "named": {"alpha": 1, "beta": "two"},
    }
    assert repr(Pipeline([operator]).describe()) == "Pipeline[\n  0:VariadicArgsOp('first', 'second', alpha=1, beta='two')\n]"


def test_describe_prints_description_once_for_top_level_call(capsys):
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe()

    captured = capsys.readouterr()

    assert captured.out == repr(description) + "\n"


def test_describe_renders_context_operators_with_captured_args():
    description = Pipeline([Store("saved"), Recall("saved")]).describe()

    assert [step.operator_args for step in description.steps] == [
        {"name": "saved"},
        {"name": "saved"},
    ]
    assert repr(description) == "Pipeline[\n  0:Store(name='saved')\n  1:Recall(name='saved')\n]"


def test_describe_keeps_batch_region_operators_in_chain_order():
    description = Pipeline([Batch(size=2), ListIdentity(), UnBatch()]).describe()

    assert [(step.label, step.kind) for step in description.steps] == [
        ("Batch", "operator"),
        ("ListIdentity", "operator"),
        ("UnBatch", "operator"),
    ]
    assert repr(description) == "Pipeline[\n  0:Batch(size=2)\n  1:ListIdentity()\n  2:UnBatch()\n]"


def test_describe_keeps_scatter_region_operators_in_chain_order():
    description = Pipeline([Scatter(max_concurrency=1), ConfiguredIntToString(), Gather()]).describe()

    assert [(step.label, step.kind) for step in description.steps] == [
        ("Scatter", "operator"),
        ("ConfiguredIntToString", "operator"),
        ("Gather", "operator"),
    ]
    assert repr(description) == (
        "Pipeline[\n"
        "  0:Scatter(max_concurrency=1)\n"
        "  1:ConfiguredIntToString()\n"
        "  2:Gather()\n"
        "]"
    )


def test_describe_expands_embedded_pipeline_by_default():
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe()

    assert len(description.steps) == 2
    assert description.steps[0].label == "Embed"
    assert description.steps[0].kind == "pipeline"
    assert [child.label for child in description.steps[0].children] == [
        "ConfiguredIntToString",
        "StringToFloat",
    ]
    assert repr(description) == (
        "Pipeline[\n"
        "  0:Embed()\n"
        "    0:ConfiguredIntToString()\n"
        "    1:StringToFloat()\n"
        "  1:FloatToBool()\n"
        "]"
    )


def test_describe_can_collapse_embedded_pipeline():
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe(expand_embedded=False)

    assert description.steps[0].children == []
    assert repr(description) == (
        "Pipeline[\n"
        "  0:Embed()\n"
        "  1:FloatToBool()\n"
        "]"
    )


def test_describe_ignores_custom_operator_labels_when_present():
    pipeline = Pipeline(
        [ConfiguredIntToString(), StringToFloat()],
        tracing=TracingConfig(collector=_Capture(), operator_labels=["to_str", "to_float"]),
    )

    description = pipeline.describe()

    assert [step.label for step in description.steps] == ["ConfiguredIntToString", "StringToFloat"]


def test_describe_preserves_extract_constructor_config():
    description = Pipeline([Extract("output0", as_="preds")]).describe()

    assert description.steps[0].operator_args == {
        "names": ("output0",),
        "as_": "preds",
    }
    assert repr(description) == "Pipeline[\n  0:Extract('output0', as_='preds')\n]"


def test_describe_preserves_explicit_none_constructor_args():
    description = Pipeline([Extract("output0", as_=None)]).describe()

    assert description.steps[0].operator_args == {
        "names": ("output0",),
        "as_": None,
    }
    assert repr(description) == "Pipeline[\n  0:Extract('output0', as_=None)\n]"


def test_describe_captures_wrapper_constructor_args():
    assert Pipeline([FilterPredictionsByClass({0, 2})]).describe().steps[0].operator_args == {
        "classes": {0, 2},
    }
    assert repr(Pipeline([FilterPredictionsByClass({0, 2})]).describe()) == (
        "Pipeline[\n  0:FilterPredictionsByClass(classes={0, 2})\n]"
    )

    assert Pipeline([FilterPredictionsByScore(0.7)]).describe().steps[0].operator_args == {
        "min_score": 0.7,
    }
    assert repr(Pipeline([FilterPredictionsByScore(0.7)]).describe()) == (
        "Pipeline[\n  0:FilterPredictionsByScore(min_score=0.7)\n]"
    )

    assert Pipeline([FilterTensorsByScore("boxes", score="scores", min_score=0.75)]).describe().steps[0].operator_args == {
        "srcs": ("boxes",),
        "score": "scores",
        "min_score": 0.75,
    }
    assert repr(Pipeline([FilterTensorsByScore("boxes", score="scores", min_score=0.75)]).describe()) == (
        "Pipeline[\n  0:FilterTensorsByScore('boxes', score='scores', min_score=0.75)\n]"
    )

    assert Pipeline([AsType("float16")]).describe().steps[0].operator_args == {
        "dtype": "float16",
    }
    assert repr(Pipeline([AsType("float16")]).describe()) == "Pipeline[\n  0:AsType(dtype='float16')\n]"


def test_describe_undecorated_custom_operator_shows_no_args():
    description = Pipeline([LegacyConfiguredIntToString(prefix="id:")]).describe()

    assert description.steps[0].operator_args == {}
    assert repr(description) == "Pipeline[\n  0:LegacyConfiguredIntToString()\n]"
