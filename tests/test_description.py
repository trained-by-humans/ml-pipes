from __future__ import annotations

from typing import Any

from ml_pipes import (
    Batch,
    Extract,
    Gather,
    ImagePayload,
    Infer,
    InvocationTrace,
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


class _Capture(TraceCollector):
    def on_trace(self, trace: InvocationTrace) -> None:
        del trace


class _FakeSession:
    def __init__(self, providers: tuple[str, ...]) -> None:
        self._providers = providers

    def get_providers(self) -> tuple[str, ...]:
        return self._providers


class _FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeLock:
    def acquire(self) -> None:
        return None


def test_describe_flat_typed_pipeline_uses_static_signatures_and_config():
    description = Pipeline([ConfiguredIntToString(prefix="id:"), StringToFloat()]).describe()

    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_config)
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
    assert "Cfg" in text
    assert "0:ConfiguredIntToString" in text
    assert "1:StringToFloat" in text
    assert "{'prefix': 'id:'}" in text
    assert "operator" not in text


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
    assert description.steps[0].operator_config == {"name": "saved"}
    assert description.steps[1].operator_config == {"name": "saved"}


def test_describe_always_groups_batch_region():
    description = Pipeline([Batch(size=2), ListIdentity(), UnBatch()]).describe()

    assert len(description.steps) == 1

    region = description.steps[0]
    assert region.label == "0:Batch"
    assert region.kind == "region"
    assert region.input_type is Any
    assert region.output_type is Any
    assert region.operator_config == {"size": 2, "timeout": 0.05}
    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_config)
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
    assert region.operator_config == {"max_concurrency": 1}
    assert [
        (step.label, step.input_type, step.output_type, step.operator_config)
        for step in region.children
    ] == [
        ("1:ConfiguredIntToString", int, str, {"prefix": ""}),
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
    assert embedded.operator_config == {}
    assert [
        (step.label, step.kind, step.input_type, step.output_type, step.operator_config)
        for step in embedded.children
    ] == [
        ("0:ConfiguredIntToString", "operator", int, str, {"prefix": ""}),
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

    assert description.steps[0].operator_config == {
        "names": "output0",
        "as_": "preds",
    }


def test_describe_preserves_infer_constructor_config_from_normalized_state():
    infer = object.__new__(Infer)
    infer.model_path = "model.onnx"
    infer.session = _FakeSession(("CPUExecutionProvider",))
    infer.input_name = "images"
    infer.input_layout = "NCHW"
    infer.model_dtype = _FakeDType("float16")
    infer.output_layouts = ("NCHW",)
    infer._lock = _FakeLock()
    infer.output_names = ("output0",)

    description = Pipeline([infer]).describe()

    assert description.steps[0].operator_config == {
        "model_path": "model.onnx",
        "providers": ("CPUExecutionProvider",),
        "input_name": "images",
        "input_layout": "NCHW",
        "dtype": "float16",
        "output_layouts": ("NCHW",),
        "serialize": True,
    }
