from __future__ import annotations

from ml_pipes.core import Pipeline


class MakeGenerator:
    def __call__(self, count: int):
        return (index for index in range(count))


class CollectGenerator:
    def __call__(self, values):
        return list(values)


def test_inspect_handles_generator_outputs_without_truncating_pipeline():
    result = Pipeline([MakeGenerator(), CollectGenerator()]).inspect(3)

    assert [span.label for span in result.spans] == ["0:MakeGenerator", "1:CollectGenerator"]
    assert not any(span.error for span in result.spans)
    assert isinstance(result.spans[0].output_value, str)
    assert "generator object" in result.spans[0].output_value
    assert result.spans[1].output_value == [0, 1, 2]
