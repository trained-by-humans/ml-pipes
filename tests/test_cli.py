from __future__ import annotations

import sys
import types

import pytest

from ml_pipes.factory import (
    _DATA_FACTORY_ATTR,
    _PIPELINE_FACTORY_ATTR,
    data_factory,
    discover_factory,
    pipeline_factory,
)
from ml_pipes.__main__ import (
    CLIError,
    _build_file_input_fns,
    _build_parser,
    _parse_axis_spec,
    _parse_axis_value,
    _parse_configs,
    cmd_benchmark,
)


# ---------------------------------------------------------------------------
# Decorator markers
# ---------------------------------------------------------------------------

def test_pipeline_factory_sets_attribute():
    def my_fn(model_path): pass
    wrapped = pipeline_factory(my_fn)
    assert getattr(wrapped, _PIPELINE_FACTORY_ATTR) is True


def test_data_factory_sets_attribute():
    def my_fn(): pass
    wrapped = data_factory(my_fn)
    assert getattr(wrapped, _DATA_FACTORY_ATTR) is True


def test_pipeline_factory_preserves_name():
    @pipeline_factory
    def my_named_fn(x): pass
    assert my_named_fn.__name__ == "my_named_fn"


def test_pipeline_factory_preserves_wrapped():
    def original(x): pass
    wrapped = pipeline_factory(original)
    assert wrapped.__wrapped__ is original


def test_decorators_exported_from_init():
    from ml_pipes import pipeline_factory as pf, data_factory as df
    assert callable(pf)
    assert callable(df)


# ---------------------------------------------------------------------------
# Axis value parsing
# ---------------------------------------------------------------------------

def test_parse_axis_value_nxm_tuple():
    assert _parse_axis_value("320x320") == (320, 320)
    assert _parse_axis_value("40x40") == (40, 40)


def test_parse_axis_value_3tuple():
    assert _parse_axis_value("1x2x3") == (1, 2, 3)


def test_parse_axis_value_int():
    assert _parse_axis_value("4") == 4


def test_parse_axis_value_float():
    assert _parse_axis_value("0.25") == pytest.approx(0.25)
    assert _parse_axis_value("1.5") == pytest.approx(1.5)


def test_parse_axis_value_str():
    assert _parse_axis_value("fast") == "fast"
    assert _parse_axis_value("true") == "true"


def test_parse_axis_spec_integers():
    key, vals = _parse_axis_spec("workers=1,2,4,8")
    assert key == "workers"
    assert vals == [1, 2, 4, 8]


def test_parse_axis_spec_tuples():
    key, vals = _parse_axis_spec("slice_wh=320x320,480x480")
    assert key == "slice_wh"
    assert vals == [(320, 320), (480, 480)]


def test_parse_axis_spec_floats():
    key, vals = _parse_axis_spec("conf=0.1,0.25,0.5")
    assert key == "conf"
    assert vals == pytest.approx([0.1, 0.25, 0.5])


def test_parse_axis_spec_no_equals_raises():
    with pytest.raises(CLIError, match="--axis must be in the form"):
        _parse_axis_spec("noequalssign")


def test_parse_axis_spec_empty_values_raises():
    with pytest.raises(CLIError, match="--axis has no values"):
        _parse_axis_spec("key=")


# ---------------------------------------------------------------------------
# Config JSON parsing
# ---------------------------------------------------------------------------

def test_parse_configs_valid():
    result = _parse_configs(['{"a": 1}', '{"b": "x"}'])
    assert result == [{"a": 1}, {"b": "x"}]


def test_parse_configs_invalid_json_raises():
    with pytest.raises(CLIError, match="invalid JSON in --config #1"):
        _parse_configs(["{bad json}"])


def test_parse_configs_non_dict_raises():
    with pytest.raises(CLIError, match="must be a JSON object"):
        _parse_configs(["[1, 2, 3]"])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _fake_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_discover_factory_single_found():
    @pipeline_factory
    def my_pf(x): pass
    m = _fake_module("_test", my_pf=my_pf, other=lambda: None)
    result = discover_factory(m, None, _PIPELINE_FACTORY_ATTR, "pipeline")
    assert result is my_pf


def test_discover_factory_multiple_raises():
    @pipeline_factory
    def pf1(x): pass
    @pipeline_factory
    def pf2(x): pass
    m = _fake_module("_test2", pf1=pf1, pf2=pf2)
    with pytest.raises(ValueError, match="multiple @pipeline_factory"):
        discover_factory(m, None, _PIPELINE_FACTORY_ATTR, "pipeline")


def test_discover_factory_none_when_absent():
    m = _fake_module("_test3", other=lambda: None)
    result = discover_factory(m, None, _PIPELINE_FACTORY_ATTR, "pipeline")
    assert result is None


def test_discover_factory_explicit_bypasses_scan():
    def explicit(config): pass
    m = _fake_module("_test4")
    result = discover_factory(m, explicit, _PIPELINE_FACTORY_ATTR, "pipeline")
    # Undecorated explicit refs are wrapped — __wrapped__ points to the original
    assert result.__wrapped__ is explicit


def test_discover_factory_explicit_undecorated_wraps_for_dict_call():
    def plain(x=1, y=2): return (x, y)
    m = _fake_module("_test5")
    result = discover_factory(m, plain, _PIPELINE_FACTORY_ATTR, "pipeline")
    assert result({"x": 10, "y": 20}) == (10, 20)


def test_discover_factory_explicit_already_decorated_not_double_wrapped():
    @pipeline_factory
    def decorated(x=1): pass
    m = _fake_module("_test6")
    result = discover_factory(m, decorated, _PIPELINE_FACTORY_ATTR, "pipeline")
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
        with pytest.raises(CLIError, match="missing required argument"):
            cmd_benchmark(args)
    finally:
        del sys.modules["_test_bench_missing"]
