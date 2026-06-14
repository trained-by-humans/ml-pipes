from __future__ import annotations

import functools
import types
from typing import Any

import pytest

from ml_pipes import Pipeline
from ml_pipes.factory import (
    DataFactory,
    Factory,
    PipelineFactory,
    data_factory,
    pipeline_factory,
)


def _fake_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _passthrough_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Decorators and wrapper behavior
# ---------------------------------------------------------------------------

def test_pipeline_factory_returns_discoverable_factory():
    def my_fn(model_path):
        return Pipeline([])

    wrapped = pipeline_factory(my_fn)
    module = _fake_module("_test_pipeline_factory", wrapped=wrapped)
    assert isinstance(wrapped, PipelineFactory)
    assert wrapped is not my_fn
    assert wrapped.__name__ == "my_fn"
    assert PipelineFactory.discover(module) is wrapped
    assert isinstance(wrapped.build({"model_path": "x"}), Pipeline)


def test_data_factory_returns_discoverable_factory():
    def my_fn():
        return lambda: ("id", "value", None, None)

    wrapped = data_factory(my_fn)
    module = _fake_module("_test_data_factory", wrapped=wrapped)
    assert isinstance(wrapped, DataFactory)
    assert wrapped is not my_fn
    assert wrapped.__name__ == "my_fn"
    assert DataFactory.discover(module) is wrapped
    assert callable(wrapped.build({}))


def test_pipeline_factory_preserves_name():
    @pipeline_factory
    def my_named_fn(x): pass

    assert my_named_fn.__name__ == "my_named_fn"


def test_pipeline_factory_preserves_direct_call():
    seen = {}

    @pipeline_factory
    def original(x):
        seen["x"] = x
        return Pipeline([])

    assert isinstance(original(2), Pipeline)
    assert seen["x"] == 2


def test_data_factory_preserves_direct_call():
    seen = {}

    @data_factory
    def original(x):
        seen["x"] = x
        return lambda: ("id", x + 1, None, None)

    assert callable(original(2))
    assert seen["x"] == 2


def test_factory_from_callable_wraps_plain_callable_for_config_dict_call():
    def plain(x=1, y=2):
        return (x, y)

    wrapped = Factory.from_callable(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped(x=10, y=20) == (10, 20)
    assert wrapped.build({"x": 10, "y": 20}) == (10, 20)


def test_factory_from_callable_passes_dict_values_as_keyword_args():
    def plain(labels: dict):
        return labels["x"]

    wrapped = Factory.from_callable(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped(labels={"x": 10}) == 10
    assert wrapped.build({"labels": {"x": 10}}) == 10


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


def test_pipeline_factory_from_callable_rejects_wrapped_decorated_factory():
    @pipeline_factory
    def decorated(x=1):
        return Pipeline([])

    wrapped = _passthrough_wrapper(decorated)

    with pytest.raises(TypeError, match=r"@pipeline_factory must be the outermost decorator"):
        PipelineFactory.from_callable(wrapped)


def test_data_factory_from_callable_rejects_wrapped_decorated_factory():
    @data_factory
    def decorated(x=1):
        return lambda: ("id", x, None, None)

    wrapped = _passthrough_wrapper(decorated)

    with pytest.raises(TypeError, match=r"@data_factory must be the outermost decorator"):
        DataFactory.from_callable(wrapped)


def test_factory_ensure_factory_wraps_plain_callable():
    def plain(value=1):
        return value

    wrapped = Factory.ensure_factory(plain)
    assert isinstance(wrapped, Factory)
    assert wrapped.build({"value": 10}) == 10


def test_factory_ensure_factory_is_idempotent():
    @pipeline_factory
    def decorated(x=1):
        return Pipeline([])

    assert Factory.ensure_factory(decorated) is decorated


def test_pipeline_factory_ensure_factory_promotes_base_factory():
    def plain(value=1):
        return Pipeline([])

    wrapped = Factory.from_callable(plain)
    promoted = PipelineFactory.ensure_factory(wrapped)
    assert isinstance(promoted, PipelineFactory)
    assert isinstance(promoted.build({"value": 10}), Pipeline)


def test_pipeline_factory_build_validates_config():
    @pipeline_factory
    def build_pipeline(workers: int) -> Pipeline[Any, Any]:
        return Pipeline([])

    with pytest.raises(TypeError, match="pipeline factory is missing required config key.*workers"):
        build_pipeline.build({})


def test_data_factory_build_validates_config():
    @data_factory
    def build_data(value: int):
        return lambda: ("id", value, None, None)

    with pytest.raises(TypeError, match="data factory is missing required config key.*value"):
        build_data.build({})


def test_pipeline_factory_validates_return_type_on_direct_call():
    @pipeline_factory
    def bad_pipeline():
        return 1

    with pytest.raises(TypeError, match="pipeline factory must return a Pipeline"):
        bad_pipeline()


def test_data_factory_validates_return_type_on_direct_call():
    @data_factory
    def bad_data():
        return None

    with pytest.raises(TypeError, match="data factory must return a callable InputFn"):
        bad_data()


def test_decorators_exported_from_init():
    from ml_pipes import (
        DataFactory as data_factory_type,
        Factory as factory_type,
        PipelineFactory as pipeline_factory_type,
        data_factory as df,
        pipeline_factory as pf,
    )

    assert callable(pf)
    assert callable(df)
    assert callable(factory_type)
    assert callable(pipeline_factory_type)
    assert callable(data_factory_type)
    assert callable(factory_type.from_callable)
    assert callable(factory_type.ensure_factory)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_pipeline_factory_single_found():
    @pipeline_factory
    def my_pf(x):
        return Pipeline([])

    module = _fake_module("_test", my_pf=my_pf, other=lambda: None)
    result = PipelineFactory.discover(module)
    assert result is my_pf


def test_discover_pipeline_factory_multiple_raises():
    @pipeline_factory
    def pf1(x): pass

    @pipeline_factory
    def pf2(x): pass

    module = _fake_module("_test2", pf1=pf1, pf2=pf2)
    with pytest.raises(ValueError, match="multiple @pipeline_factory"):
        PipelineFactory.discover(module)


def test_discover_pipeline_factory_none_when_absent():
    module = _fake_module("_test3", other=lambda: None)
    result = PipelineFactory.discover(module)
    assert result is None


def test_discover_pipeline_and_data_factory_use_distinct_classes():
    @pipeline_factory
    def my_pipeline(x=1):
        return Pipeline([])

    @data_factory
    def my_data(x=1):
        return lambda: ("id", x, None, None)

    module = _fake_module("_test_kinds", my_pipeline=my_pipeline, my_data=my_data)
    assert PipelineFactory.discover(module) is my_pipeline
    assert DataFactory.discover(module) is my_data


def test_discover_pipeline_factory_explicit_wraps_dict_valued_callable():
    seen = {}

    def explicit(labels: dict):
        seen["labels"] = labels
        return Pipeline([])

    module = _fake_module("_test4")
    result = PipelineFactory.discover(module, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert isinstance(result.build({"labels": {"x": 1}}), Pipeline)
    assert seen["labels"] == {"x": 1}


def test_discover_pipeline_factory_explicit_wraps_keyword_callable():
    seen = {}

    def explicit(x=1, y=2):
        seen["args"] = (x, y)
        return Pipeline([])

    module = _fake_module("_test4_kwargs")
    result = PipelineFactory.discover(module, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert isinstance(result(x=3, y=4), Pipeline)
    assert seen["args"] == (3, 4)
    assert isinstance(result.build({"x": 3, "y": 4}), Pipeline)
    assert seen["args"] == (3, 4)


def test_discover_pipeline_factory_explicit_already_decorated_not_double_wrapped():
    @pipeline_factory
    def decorated(x=1): pass

    module = _fake_module("_test6")
    result = PipelineFactory.discover(module, decorated)
    assert result is decorated


def test_discover_pipeline_factory_wrapped_factory_raises():
    @pipeline_factory
    def decorated(x=1):
        return Pipeline([])

    wrapped = _passthrough_wrapper(decorated)
    module = _fake_module("_wrapped_pipeline_factory", wrapped=wrapped)

    with pytest.raises(TypeError, match=r"wraps @pipeline_factory"):
        PipelineFactory.discover(module)


def test_discover_data_factory_wrapped_factory_raises():
    @data_factory
    def decorated(x=1):
        return lambda: ("id", x, None, None)

    wrapped = _passthrough_wrapper(decorated)
    module = _fake_module("_wrapped_data_factory", wrapped=wrapped)

    with pytest.raises(TypeError, match=r"wraps @data_factory"):
        DataFactory.discover(module)
