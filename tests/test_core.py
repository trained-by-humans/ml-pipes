from ml_pipes import Context, Pipeline, Value


def test_context_add_returns_new_context():
    context = Context()
    next_context = context.add("resize")

    assert context.transforms == ()
    assert context.metadata == {}
    assert next_context.transforms == ("resize",)
    assert next_context.metadata == {}


def test_pipeline_applies_operators_in_order():
    pipeline = Pipeline(
        [
            lambda value: value + 2,
            lambda value: value * 3,
        ]
    )

    assert pipeline(4) == 18


def test_value_default_context():
    value = Value(data="image")

    assert value.context.transforms == ()
    assert value.context.metadata == {}
