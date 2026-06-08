from __future__ import annotations

import sys
import types

import pytest

from ml_pipes.factory import (
    DataFactory,
    Factory,
    PipelineFactory,
    data_factory,
    pipeline_factory,
)
from ml_pipes.__main__ import (
    CLIError,
    _build_file_input_fns,
    _build_parser,
    _parse_config_arg,
    _parse_config_axis,
    _parse_config_value,
    _parse_config_list,
    _resolve_pipeline_factory,
    cmd_benchmark,
)


# ---------------------------------------------------------------------------
# Decorator markers
# ---------------------------------------------------------------------------

def test_pipeline_factory_returns_discoverable_factory():
    def my_fn(model_path): pass
    wrapped = pipeline_factory(my_fn)
    module = _fake_module("_test_pipeline_factory", wrapped=wrapped)
    assert isinstance(wrapped, PipelineFactory)
    assert wrapped is not my_fn
    assert wrapped.__name__ == "my_fn"
    assert PipelineFactory.discover(module) is wrapped
    assert wrapped.from_config({"model_path": "x"}) is None


def test_data_factory_returns_discoverable_factory():
    def my_fn(): pass
    wrapped = data_factory(my_fn)
    module = _fake_module("_test_data_factory", wrapped=wrapped)
    assert isinstance(wrapped, DataFactory)
    assert wrapped is not my_fn
    assert wrapped.__name__ == "my_fn"
    assert DataFactory.discover(module) is wrapped
    assert wrapped.from_config({}) is None


def test_pipeline_factory_preserves_name():
    @pipeline_factory
    def my_named_fn(x): pass
    assert my_named_fn.__name__ == "my_named_fn"


def test_pipeline_factory_preserves_direct_call():
    @pipeline_factory
    def original(x):
        return x + 1

    assert original(2) == 3


def test_data_factory_preserves_direct_call():
    @data_factory
    def original(x):
        return x + 1

    assert original(2) == 3


def test_factory_from_callable_wraps_plain_callable_for_config_dict_call():
    def plain(x=1, y=2):
        return (x, y)

    wrapped = Factory.from_callable(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped(x=10, y=20) == (10, 20)
    assert wrapped.from_config({"x": 10, "y": 20}) == (10, 20)


def test_factory_from_callable_passes_dict_values_as_keyword_args():
    def plain(labels: dict):
        return labels["x"]

    wrapped = Factory.from_callable(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped(labels={"x": 10}) == 10
    assert wrapped.from_config({"labels": {"x": 10}}) == 10


def test_factory_from_callable_rejects_existing_factory():
    @pipeline_factory
    def decorated(x=1):
        return x

    with pytest.raises(TypeError, match="Factory.ensure_factory"):
        Factory.from_callable(decorated)


def test_pipeline_factory_from_callable_rejects_other_factory_subclass():
    @data_factory
    def decorated(x=1):
        return x

    with pytest.raises(TypeError, match="PipelineFactory.ensure_factory"):
        PipelineFactory.from_callable(decorated)


def test_data_factory_from_callable_rejects_other_factory_subclass():
    @pipeline_factory
    def decorated(x=1):
        return x

    with pytest.raises(TypeError, match="DataFactory.ensure_factory"):
        DataFactory.from_callable(decorated)


def test_factory_ensure_factory_wraps_plain_callable():
    def plain(value=1):
        return value

    wrapped = Factory.ensure_factory(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped.from_config({"value": 10}) == 10


def test_factory_ensure_factory_is_idempotent():
    @pipeline_factory
    def decorated(x=1):
        return x

    assert Factory.ensure_factory(decorated) is decorated


def test_pipeline_factory_ensure_factory_promotes_base_factory():
    def plain(value=1):
        return value

    wrapped = Factory.from_callable(plain)
    promoted = PipelineFactory.ensure_factory(wrapped)
    assert isinstance(promoted, PipelineFactory)
    assert promoted.from_config({"value": 10}) == 10


def test_decorators_exported_from_init():
    from ml_pipes import (
        DataFactory as data_factory_type,
        Factory as factory_type,
        PipelineFactory as pipeline_factory_type,
        pipeline_factory as pf,
        data_factory as df,
    )
    assert callable(pf)
    assert callable(df)
    assert callable(factory_type)
    assert callable(pipeline_factory_type)
    assert callable(data_factory_type)
    assert callable(factory_type.from_callable)
    assert callable(factory_type.ensure_factory)


# ---------------------------------------------------------------------------
# Axis value parsing
# ---------------------------------------------------------------------------

def test_parse_config_value_nxm_tuple():
    assert _parse_config_value("320x320") == (320, 320)
    assert _parse_config_value("40x40") == (40, 40)


def test_parse_config_value_3tuple():
    assert _parse_config_value("1x2x3") == (1, 2, 3)


def test_parse_config_value_int():
    assert _parse_config_value("4") == 4


def test_parse_config_value_float():
    assert _parse_config_value("0.25") == pytest.approx(0.25)
    assert _parse_config_value("1.5") == pytest.approx(1.5)


def test_parse_config_value_str():
    assert _parse_config_value("fast") == "fast"
    assert _parse_config_value("true") == "true"


def test_parse_config_axis_integers():
    key, vals = _parse_config_axis("workers=1,2,4,8")
    assert key == "workers"
    assert vals == [1, 2, 4, 8]


def test_parse_config_axis_tuples():
    key, vals = _parse_config_axis("slice_wh=320x320,480x480")
    assert key == "slice_wh"
    assert vals == [(320, 320), (480, 480)]


def test_parse_config_axis_floats():
    key, vals = _parse_config_axis("conf=0.1,0.25,0.5")
    assert key == "conf"
    assert vals == pytest.approx([0.1, 0.25, 0.5])


def test_parse_config_axis_no_equals_raises():
    with pytest.raises(CLIError, match="--axis must be in the form"):
        _parse_config_axis("noequalssign")


def test_parse_config_axis_empty_values_raises():
    with pytest.raises(CLIError, match="--axis has no values"):
        _parse_config_axis("key=")


# ---------------------------------------------------------------------------
# Config JSON parsing
# ---------------------------------------------------------------------------

def test_parse_config_list_valid():
    result = _parse_config_list(['{"a": 1}', '{"b": "x"}'])
    assert result == [{"a": 1}, {"b": "x"}]


def test_parse_config_list_invalid_json_raises():
    with pytest.raises(CLIError, match="invalid JSON in --config #1"):
        _parse_config_list(["{bad json}"])


def test_parse_config_list_non_dict_raises():
    with pytest.raises(CLIError, match="must be a JSON object"):
        _parse_config_list(["[1, 2, 3]"])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _fake_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_discover_pipeline_factory_single_found():
    @pipeline_factory
    def my_pf(x): pass
    m = _fake_module("_test", my_pf=my_pf, other=lambda: None)
    result = PipelineFactory.discover(m)
    assert result is my_pf


def test_discover_pipeline_factory_multiple_raises():
    @pipeline_factory
    def pf1(x): pass
    @pipeline_factory
    def pf2(x): pass
    m = _fake_module("_test2", pf1=pf1, pf2=pf2)
    with pytest.raises(ValueError, match="multiple @pipeline_factory"):
        PipelineFactory.discover(m)


def test_discover_pipeline_factory_none_when_absent():
    m = _fake_module("_test3", other=lambda: None)
    result = PipelineFactory.discover(m)
    assert result is None


def test_discover_pipeline_and_data_factory_use_distinct_classes():
    @pipeline_factory
    def my_pipeline(x=1):
        return x

    @data_factory
    def my_data(x=1):
        return x

    m = _fake_module("_test_kinds", my_pipeline=my_pipeline, my_data=my_data)
    assert PipelineFactory.discover(m) is my_pipeline
    assert DataFactory.discover(m) is my_data


def test_discover_pipeline_factory_explicit_wraps_dict_valued_callable():
    def explicit(labels: dict):
        return labels["x"]

    m = _fake_module("_test4")
    result = PipelineFactory.discover(m, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert result.from_config({"labels": {"x": 1}}) == 1


def test_discover_pipeline_factory_explicit_wraps_keyword_callable():
    def explicit(x=1, y=2):
        return (x, y)

    m = _fake_module("_test4_kwargs")
    result = PipelineFactory.discover(m, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert result(x=3, y=4) == (3, 4)
    assert result.from_config({"x": 3, "y": 4}) == (3, 4)


def test_resolve_pipeline_factory_explicit_undecorated_wraps_keyword_callable():
    def plain(x=1, y=2): return (x, y)
    m = _fake_module("_test5")
    result = _resolve_pipeline_factory(m, plain, "_test5:plain")
    assert isinstance(result, PipelineFactory)
    assert result(x=10, y=20) == (10, 20)
    assert result.from_config({"x": 10, "y": 20}) == (10, 20)


def test_discover_pipeline_factory_explicit_already_decorated_not_double_wrapped():
    @pipeline_factory
    def decorated(x=1): pass
    m = _fake_module("_test6")
    result = PipelineFactory.discover(m, decorated)
    assert result is decorated


# ---------------------------------------------------------------------------
# File input builder
# ---------------------------------------------------------------------------

def test_build_file_input_fns(tmp_path):
    f1 = tmp_path / "a.jpg"
    f2 = tmp_path / "b.jpg"
    f1.write_bytes(b"fake")
    f2.write_bytes(b"fake")
    fns, labels = _build_file_input_fns([str(f1), str(f2)])
    assert len(fns) == 2
    assert labels == ["a.jpg", "b.jpg"]
    id1, val1, tag1, meta1 = fns[0]()
    assert id1 == "a.jpg"
    assert val1 == f1
    assert tag1 is None
    assert meta1 is None


def test_build_file_input_fns_basename_collision(tmp_path):
    d1 = tmp_path / "setA"
    d2 = tmp_path / "setB"
    d1.mkdir()
    d2.mkdir()
    f1 = d1 / "img001.jpg"
    f2 = d2 / "img001.jpg"
    f1.write_bytes(b"fake")
    f2.write_bytes(b"fake")
    fns, labels = _build_file_input_fns([str(f1), str(f2)])
    assert labels[0] != labels[1], "colliding basenames must produce distinct labels"
    assert str(f1) == labels[0]
    assert str(f2) == labels[1]
    id1, _, _, _ = fns[0]()
    id2, _, _, _ = fns[1]()
    assert id1 != id2


def test_build_file_input_fns_missing_raises(tmp_path):
    with pytest.raises(CLIError, match="input file not found"):
        _build_file_input_fns([str(tmp_path / "nonexistent.jpg")])


# ---------------------------------------------------------------------------
# Integration — cmd_benchmark with injected fake module
# ---------------------------------------------------------------------------

class _Identity:
    def __call__(self, x):
        return x


def _make_identity_pipeline(**kwargs):
    from ml_pipes import Pipeline
    return Pipeline([_Identity()])


_wrapped_identity = pipeline_factory(_make_identity_pipeline)


def test_cmd_benchmark_end_to_end(tmp_path, capsys):
    f = tmp_path / "input.bin"
    f.write_bytes(b"data")

    mod = types.ModuleType("_test_bench_integration")
    mod._wrapped_identity = _wrapped_identity
    sys.modules["_test_bench_integration"] = mod

    try:
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "_test_bench_integration:_wrapped_identity",
            "--input", str(f),
            "--runs", "2", "--warmup", "1",
        ])
        code = cmd_benchmark(args)
        assert code == 0
        out = capsys.readouterr().out
        assert "total" in out
        assert "mean" in out
    finally:
        del sys.modules["_test_bench_integration"]


def test_cmd_benchmark_missing_required_arg_raises(tmp_path):
    f = tmp_path / "input.bin"
    f.write_bytes(b"data")

    def _needs_arg(required_param):  # no default → must be in config
        from ml_pipes import Pipeline
        return Pipeline([_Identity()])

    wrapped = pipeline_factory(_needs_arg)

    mod = types.ModuleType("_test_bench_missing")
    mod.wrapped = wrapped
    sys.modules["_test_bench_missing"] = mod

    try:
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "_test_bench_missing:wrapped",
            "--input", str(f),
            "--runs", "2", "--warmup", "1",
        ])
        with pytest.raises(TypeError):
            cmd_benchmark(args)
    finally:
        del sys.modules["_test_bench_missing"]


# ---------------------------------------------------------------------------
# --arg / --data-arg parsing
# ---------------------------------------------------------------------------

def test_parse_config_arg_valid():
    assert _parse_config_arg("workers=4") == ("workers", 4)
    assert _parse_config_arg("mode=fast") == ("mode", "fast")
    assert _parse_config_arg("wh=320x240") == ("wh", (320, 240))


def test_parse_config_arg_no_equals_raises():
    with pytest.raises(CLIError, match="--arg must be in the form"):
        _parse_config_arg("noequalssign")


def test_parse_config_arg_empty_key_raises():
    with pytest.raises(CLIError, match="--arg key is empty"):
        _parse_config_arg("=value")



# ---------------------------------------------------------------------------
# Parser: --data-arg present on run and benchmark
# ---------------------------------------------------------------------------

def test_parser_data_arg_on_run():
    parser = _build_parser()
    args = parser.parse_args(["run", "some.module", "--data-arg", "image_path=img.jpg"])
    assert args.data_args == ["image_path=img.jpg"]


def test_parser_data_arg_on_benchmark():
    parser = _build_parser()
    args = parser.parse_args([
        "benchmark", "some.module",
        "--data-arg", "image_path=img.jpg",
        "--runs", "2",
    ])
    assert args.data_args == ["image_path=img.jpg"]


# ---------------------------------------------------------------------------
# Parser: benchmark mutual exclusion groups
# ---------------------------------------------------------------------------

def test_parser_benchmark_arg_and_config_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark", "some.module",
            "--arg", "workers=4",
            "--config", '{"workers": 4}',
        ])


def test_parser_benchmark_arg_and_axis_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark", "some.module",
            "--arg", "workers=4",
            "--axis", "workers=1,2,4",
        ])


def test_parser_benchmark_data_arg_and_data_config_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark", "some.module",
            "--data-arg", "image_path=img.jpg",
            "--data-config", '{"image_path": "img.jpg"}',
        ])


def test_parser_benchmark_data_arg_and_data_axis_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark", "some.module",
            "--data-arg", "image_path=img.jpg",
            "--data-axis", "image_path=img1.jpg,img2.jpg",
        ])


def test_parser_benchmark_data_config_and_data_axis_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "benchmark", "some.module",
            "--data-config", '{"image_path": "img.jpg"}',
            "--data-axis", "image_path=img1.jpg,img2.jpg",
        ])


# ---------------------------------------------------------------------------
# Integration — cmd_benchmark with data_factory
# ---------------------------------------------------------------------------

def test_cmd_benchmark_with_data_factory(tmp_path, capsys):
    @data_factory
    def make_data(image_path="default.jpg"):
        def _fn():
            return (image_path, image_path, None, None)
        return _fn

    mod = types.ModuleType("_test_sweep_data")
    mod._wrapped_identity = _wrapped_identity
    mod.make_data = make_data
    sys.modules["_test_sweep_data"] = mod

    try:
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "_test_sweep_data:_wrapped_identity", "_test_sweep_data:make_data",
            "--data-arg", "image_path=test.jpg",
            "--runs", "2", "--warmup", "1",
        ])
        code = cmd_benchmark(args)
        assert code == 0
        out = capsys.readouterr().out
        assert "mean" in out
    finally:
        del sys.modules["_test_sweep_data"]


def test_cmd_benchmark_with_data_axis(tmp_path, capsys):
    @data_factory
    def make_data(image_path="default.jpg"):
        def _fn():
            return (image_path, image_path, None, None)
        return _fn

    mod = types.ModuleType("_test_sweep_data_axis")
    mod._wrapped_identity = _wrapped_identity
    mod.make_data = make_data
    sys.modules["_test_sweep_data_axis"] = mod

    try:
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "_test_sweep_data_axis:_wrapped_identity", "_test_sweep_data_axis:make_data",
            "--data-axis", "image_path=img1.jpg,img2.jpg",
            "--runs", "2", "--warmup", "1",
        ])
        code = cmd_benchmark(args)
        assert code == 0
        out = capsys.readouterr().out
        assert "mean" in out
    finally:
        del sys.modules["_test_sweep_data_axis"]
