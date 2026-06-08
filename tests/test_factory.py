from __future__ import annotations

import types

import pytest

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


# ---------------------------------------------------------------------------
# Decorators and wrapper behavior
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
    def my_pf(x): pass

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
        return x

    @data_factory
    def my_data(x=1):
        return x

    module = _fake_module("_test_kinds", my_pipeline=my_pipeline, my_data=my_data)
    assert PipelineFactory.discover(module) is my_pipeline
    assert DataFactory.discover(module) is my_data


def test_discover_pipeline_factory_explicit_wraps_dict_valued_callable():
    def explicit(labels: dict):
        return labels["x"]

    module = _fake_module("_test4")
    result = PipelineFactory.discover(module, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert result.from_config({"labels": {"x": 1}}) == 1


def test_discover_pipeline_factory_explicit_wraps_keyword_callable():
    def explicit(x=1, y=2):
        return (x, y)

    module = _fake_module("_test4_kwargs")
    result = PipelineFactory.discover(module, explicit)
    assert isinstance(result, PipelineFactory)
    assert result is not explicit
    assert result(x=3, y=4) == (3, 4)
    assert result.from_config({"x": 3, "y": 4}) == (3, 4)


def test_discover_pipeline_factory_explicit_already_decorated_not_double_wrapped():
    @pipeline_factory
    def decorated(x=1): pass

    module = _fake_module("_test6")
    result = PipelineFactory.discover(module, decorated)
    assert result is decorated
