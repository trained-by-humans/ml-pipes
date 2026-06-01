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
    ImagePayload,
    InvocationTrace,
    Operator,
    Pipeline,
    Recall,
    ResizeTransform,
    Scatter,
    Store,
    TraceCollector,
    TracingConfig,
    UnBatch,
    embed,
)
from ml_pipes.operator import get_operator_args


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


class PairToString:
    def __call__(self, left: int, right: float) -> str:
        return f"{left}:{right}"


class UntypedOp:
    def __call__(self, value):
        return value


class BrokenHints:
    def __call__(self, value: "MissingType") -> "AlsoMissing":
        return value


class ImageOp:
    def __call__(self, image: ImagePayload) -> tuple[ImagePayload, ResizeTransform]:
        raise NotImplementedError


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


def test_describe_flat_typed_pipeline_uses_static_signatures_and_args():
    description = Pipeline([ConfiguredIntToString(prefix="id:"), StringToFloat()]).describe()

    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_args)
        for step in description.steps
    ] == [
        ("0:ConfiguredIntToString", "operator", int, str, {"prefix": "id:"}),
        ("1:StringToFloat", "operator", str, float, {}),
    ]

    text = repr(description)
    assert "PipelineDescription" in text
    assert text.startswith("PipelineDescription\n\n")
    assert "Step" in text and " | " in text
    assert "-+-" in text
    assert "Args" in text
    assert "0:ConfiguredIntToString" in text
    assert "1:StringToFloat" in text
    assert "{'prefix': 'id:'}" in text
    assert "operator" not in text


def test_describe_operator_config_aliases_operator_args():
    step = Pipeline([ConfiguredIntToString(prefix="id:")]).describe().steps[0]

    assert step.operator_config is step.operator_args


def test_get_operator_args_can_expand_defaults_on_read():
    operator = ConfiguredIntToString()

    assert get_operator_args(operator) == {}
    assert get_operator_args(operator, include_defaults=True) == {"prefix": ""}


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
        "{'flag': True, 'count': 3, 'ratio': 0.5, 'label': 'hello', "
        "'payload': b'data', 'maybe': None}"
    ) in repr(description)


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
        "{'coords': (1, 2), 'names': ['a', 'b'], 'mapping': {'left': 1, 'right': 2}, "
        "'classes': {0, 2}, 'frozen': frozenset({'alpha', 'beta'})}"
    ) in repr(description)


def test_operator_args_capture_lambda_and_describe_with_lambda_label():
    operator = CallableArgOp(_MODULE_LAMBDA)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is _MODULE_LAMBDA
    assert "'fn': <lambda>" in repr(description)


def test_operator_args_capture_function_and_describe_with_function_name():
    operator = CallableArgOp(_named_identity)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is _named_identity
    assert "'fn': _named_identity" in repr(description)


def test_operator_args_capture_callable_object_and_describe_with_type_name():
    callable_object = _CallableArg()
    operator = CallableArgOp(callable_object)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["fn"] is callable_object
    assert "'fn': _CallableArg" in repr(description)


def test_operator_args_capture_custom_object_and_describe_with_repr():
    payload = _CustomArg("example")
    operator = CustomObjectArgOp(payload)
    description = Pipeline([operator]).describe()

    assert get_operator_args(operator)["payload"] is payload
    assert "'payload': _CustomArg(name='example')" in repr(description)


def test_get_operator_args_captures_varargs_and_kwargs():
    operator = VariadicArgsOp("first", "second", alpha=1, beta="two")

    assert get_operator_args(operator) == {
        "items": ("first", "second"),
        "named": {"alpha": 1, "beta": "two"},
    }
    assert (
        "{'items': ('first', 'second'), 'named': {'alpha': 1, 'beta': 'two'}}"
        in repr(Pipeline([operator]).describe())
    )


def test_describe_prints_description_once_for_top_level_call(capsys):
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe()

    captured = capsys.readouterr()

    assert captured.out == repr(description) + "\n"
    assert captured.out.count("PipelineDescription") == 1


def test_describe_collapses_multi_argument_signatures_to_tuple_inputs():
    description = Pipeline([PairToString()]).describe()

    assert description.steps[0].input_type == tuple[int, float]
    assert description.steps[0].output_type is str
    assert "(int, float)" in repr(description)
    assert "tuple[int, float]" not in repr(description)


def test_describe_context_ops_fall_back_to_any_without_validation():
    description = Pipeline([Store("saved"), Recall("saved")]).describe()

    assert [(step.input_type, step.output_type) for step in description.steps] == [
        (Any, Any),
        (Any, Any),
    ]
    assert description.steps[0].operator_args == {"name": "saved"}
    assert description.steps[1].operator_args == {"name": "saved"}


def test_describe_always_groups_batch_region():
    description = Pipeline([Batch(size=2), ListIdentity(), UnBatch()]).describe()

    assert len(description.steps) == 1

    region = description.steps[0]
    assert region.label == "0:Batch"
    assert region.kind == "region"
    assert region.input_type is Any
    assert region.output_type is Any
    assert region.operator_args == {"size": 2}
    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_args)
        for step in region.children
    ] == [
        ("1:ListIdentity", "operator", list[int], list[int], {}),
    ]


def test_describe_always_groups_scatter_region():
    description = Pipeline([Scatter(max_concurrency=1), ConfiguredIntToString(), Gather()]).describe()

    assert len(description.steps) == 1

    region = description.steps[0]
    assert region.label == "0:Scatter"
    assert region.kind == "region"
    assert region.input_type is Any
    assert region.output_type is Any
    assert region.operator_args == {"max_concurrency": 1}
    assert [
        (step.label, step.input_type, step.output_type, step.operator_args)
        for step in region.children
    ] == [
        ("1:ConfiguredIntToString", int, str, {}),
    ]


def test_describe_expands_embedded_pipeline_by_default():
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe()

    assert len(description.steps) == 2

    embedded = description.steps[0]
    assert embedded.label == "0:Embed"
    assert embedded.kind == "pipeline"
    assert embedded.input_type is int
    assert embedded.output_type is float
    assert embedded.operator_args == {}
    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_args)
        for step in embedded.children
    ] == [
        ("0:ConfiguredIntToString", "operator", int, str, {}),
        ("1:StringToFloat", "operator", str, float, {}),
    ]

    tail = description.steps[1]
    assert tail.label == "1:FloatToBool"
    assert tail.kind == "operator"
    assert tail.input_type is float
    assert tail.output_type is bool


def test_describe_can_collapse_embedded_pipeline():
    inner = Pipeline([ConfiguredIntToString(), StringToFloat()])
    description = Pipeline([embed(inner), FloatToBool()]).describe(expand_embedded=False)

    embedded = description.steps[0]
    assert embedded.kind == "pipeline"
    assert embedded.input_type is int
    assert embedded.output_type is float
    assert embedded.children == []


def test_describe_softly_recovers_embedded_boundary_from_inner_contract():
    inner = Pipeline([Batch(size=2), ListIdentity(), UnBatch()])
    description = Pipeline([embed(inner)]).describe(expand_embedded=False)

    embedded = description.steps[0]
    assert embedded.kind == "pipeline"
    assert embedded.input_type is int
    assert embedded.output_type is int
    assert embedded.children == []


def test_describe_falls_back_to_any_for_untyped_operator():
    description = Pipeline([UntypedOp(), ConfiguredIntToString()]).describe()

    assert description.steps[0].input_type is Any
    assert description.steps[0].output_type is Any
    assert description.steps[1].input_type is int
    assert description.steps[1].output_type is str


def test_describe_falls_back_to_any_when_type_hints_cannot_be_resolved():
    description = Pipeline([BrokenHints(), ConfiguredIntToString()]).describe()

    assert description.steps[0].input_type is Any
    assert description.steps[0].output_type is Any
    assert description.steps[1].input_type is int
    assert description.steps[1].output_type is str


def test_describe_shows_local_type_names_without_module_prefixes():
    description = Pipeline([ImageOp()]).describe()

    text = repr(description)

    assert "ImagePayload" in text
    assert "ResizeTransform" in text
    assert "(ImagePayload, ResizeTransform)" in text
    assert "tuple[ImagePayload, ResizeTransform]" not in text
    assert "ml_pipes.types" not in text


def test_describe_ignores_custom_operator_labels_when_present():
    pipeline = Pipeline(
        [ConfiguredIntToString(), StringToFloat()],
        tracing=TracingConfig(collector=_Capture(), operator_labels=["to_str", "to_float"]),
    )

    description = pipeline.describe()

    assert [step.label for step in description.steps] == ["0:ConfiguredIntToString", "1:StringToFloat"]


def test_describe_leaves_unmatched_regions_as_plain_operators():
    description = Pipeline([Scatter(max_concurrency=1), ConfiguredIntToString(), UnBatch()]).describe()

    assert [step.kind for step in description.steps] == ["operator", "operator", "operator"]
    assert description.steps[0].input_type is Any
    assert description.steps[2].output_type is Any


def test_describe_preserves_extract_constructor_config():
    description = Pipeline([Extract("output0", as_="preds")]).describe()

    assert description.steps[0].operator_args == {
        "names": ("output0",),
        "as_": "preds",
    }


def test_describe_preserves_explicit_none_constructor_args():
    description = Pipeline([Extract("output0", as_=None)]).describe()

    assert description.steps[0].operator_args == {
        "names": ("output0",),
        "as_": None,
    }


def test_describe_captures_wrapper_constructor_args():
    assert Pipeline([FilterPredictionsByClass({0, 2})]).describe().steps[0].operator_args == {
        "classes": {0, 2},
    }
    assert Pipeline([FilterPredictionsByScore(0.7)]).describe().steps[0].operator_args == {
        "min_score": 0.7,
    }
    assert Pipeline([FilterTensorsByScore("boxes", score="scores", min_score=0.75)]).describe().steps[0].operator_args == {
        "srcs": ("boxes",),
        "score": "scores",
        "min_score": 0.75,
    }
    assert Pipeline([AsType("float16")]).describe().steps[0].operator_args == {
        "dtype": "float16",
    }


def test_describe_undecorated_custom_operator_shows_no_args():
    description = Pipeline([LegacyConfiguredIntToString(prefix="id:")]).describe()

    assert description.steps[0].operator_args == {}
