from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.benchmark import (
    Benchmark,
    BenchmarkBuilder,
    BenchmarkCollector,
    BenchmarkSweep,
    BenchmarkResult,
    InvocationStat,
    InvocationStatDiff,
    MeasurementConfig,
)
from ml_pipes.factory import (
    DataFactory,
    InputFn,
    PipelineFactory,
)
from ml_pipes.tracing import InvocationTrace, StepSpan
from ml_pipes.collectors import PrintCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AddOne:
    def __call__(self, x: int) -> int:
        return x + 1


class _Double:
    def __call__(self, x: int) -> int:
        return x * 2


def _static_input(value: int = 0) -> tuple[str, int, None, None]:
    return ("input", value, None, None)


def _make_pipeline() -> Pipeline[int, int]:
    return Pipeline([_AddOne(), _Double()])


# ---------------------------------------------------------------------------
# BenchmarkCollector — standalone
# ---------------------------------------------------------------------------

def test_collector_skips_warmup_runs():
    config = MeasurementConfig(runs=5, warmup=3, percentiles=(0.50,))
    collector = BenchmarkCollector(config)

    pipeline = _make_pipeline()
    pipeline.set_tracing(collector)
    for _ in range(8):  # 3 warmup + 5 measured
        pipeline(1)

    collector.stop()
    result = collector.report(label="test")
    assert result.total.count == 5


def test_collector_percentile_keys_match_config():
    config = MeasurementConfig(runs=10, warmup=2, percentiles=(0.25, 0.75, 0.90))
    collector = BenchmarkCollector(config)
    pipeline = _make_pipeline()
    pipeline.set_tracing(collector)
    for _ in range(12):
        pipeline(1)
    collector.stop()
    result = collector.report()
    assert set(result.total.percentiles.keys()) == {0.25, 0.75, 0.90}
    for span in result.operators:
        assert set(span.percentiles.keys()) == {0.25, 0.75, 0.90}


def test_collector_stddev_matches_numpy():
    config = MeasurementConfig(runs=20, warmup=0, percentiles=(0.50,))

    # Patch _collect directly to feed known durations
    collector = BenchmarkCollector(config)
    known_durations_s = [0.010 + 0.001 * i for i in range(20)]
    for d in known_durations_s:
        trace = InvocationTrace(
            spans=[StepSpan(label="0:_AddOne", start_time=0.0, duration_s=d * 0.4),
                   StepSpan(label="1:_Double", start_time=0.0, duration_s=d * 0.6)],
            total_duration_s=d,
        )
        collector._collect(trace)

    collector.stop()
    result = collector.report()
    expected_std = float(np.std([d * 1000 for d in known_durations_s], ddof=1))
    assert abs(result.total.stddev_ms - expected_std) < 1e-6


def test_collector_percentile_values_match_numpy():
    config = MeasurementConfig(runs=20, warmup=0, percentiles=(0.50, 0.95))
    collector = BenchmarkCollector(config)
    durations_s = [0.005 + 0.001 * i for i in range(20)]
    for d in durations_s:
        trace = InvocationTrace(
            spans=[StepSpan(label="0:op", start_time=0.0, duration_s=d)],
            total_duration_s=d,
        )
        collector._collect(trace)
    collector.stop()
    result = collector.report()
    arr_ms = np.array(durations_s) * 1000
    assert abs(result.total.percentiles[0.50] - float(np.percentile(arr_ms, 50))) < 1e-6
    assert abs(result.total.percentiles[0.95] - float(np.percentile(arr_ms, 95))) < 1e-6


# ---------------------------------------------------------------------------
# BenchmarkResult — save / load / to_table / to_dict
# ---------------------------------------------------------------------------

def _make_result(label: str = "test") -> BenchmarkResult:
    config = MeasurementConfig(runs=10, warmup=2, percentiles=(0.50, 0.95))
    collector = BenchmarkCollector(config)
    pipeline = _make_pipeline()
    pipeline.set_tracing(collector)
    for _ in range(12):
        pipeline(1)
    collector.stop()
    return collector.report(label=label, metadata={"env": "test"})


def test_result_to_table_renders():
    result = _make_result()
    table = result.to_table()
    assert "total" in table
    assert "p50" in table
    assert "p95" in table
    assert "mean" in table


def test_result_to_table_expand_regions_false_hides_child_spans():
    result = _make_result()
    # Inject a child stat into the first operator to simulate a Scatter region
    child_stat = InvocationStat(label="region:0", count=10, mean_ms=1.0, stddev_ms=0.1,
                                min_ms=0.8, max_ms=1.2, percentiles={0.50: 1.0, 0.95: 1.1})
    result.operators[0].children.append(child_stat)
    table = result.to_table(expand_regions=False)
    assert "total" in table
    assert "* Child spans are collapsed" in table
    assert result.operators[0].label + "*" in table
    assert "region:0" not in table


def test_result_to_table_expand_regions_true_shows_children():
    result = _make_result()
    child_stat = InvocationStat(label="region:0", count=10, mean_ms=1.0, stddev_ms=0.1,
                                min_ms=0.8, max_ms=1.2, percentiles={0.50: 1.0, 0.95: 1.1})
    result.operators[0].children.append(child_stat)
    table = result.to_table(expand_regions=True)
    assert "region:0" in table
    for op in result.operators:
        assert op.label in table


def test_result_to_table_expand_regions_false_no_children_no_note():
    result = _make_result()
    # No children — footnote should not appear
    table = result.to_table(expand_regions=False)
    assert "* Child spans are collapsed" not in table
    for op in result.operators:
        assert op.label in table


def test_result_to_dict_is_json_serializable():
    result = _make_result()
    d = result.to_dict()
    json.dumps(d)  # must not raise


def test_result_save_load_roundtrip(tmp_path):
    result = _make_result(label="roundtrip")
    result.metadata["extra"] = "value"
    path = str(tmp_path / "bench.json")
    result.save(path)
    loaded = BenchmarkResult.load(path)
    assert loaded.label == "roundtrip"
    assert loaded.metadata["extra"] == "value"
    assert loaded.total.count == result.total.count
    assert loaded.total.mean_ms == pytest.approx(result.total.mean_ms)
    assert set(loaded.total.percentiles.keys()) == set(result.total.percentiles.keys())
    assert len(loaded.operators) == len(result.operators)


# ---------------------------------------------------------------------------
# BenchmarkDiff
# ---------------------------------------------------------------------------

def test_diff_detects_delta():
    span_a = InvocationStat(label="total", count=10, mean_ms=10.0, stddev_ms=1.0,
                           min_ms=8.0, max_ms=12.0, percentiles={0.95: 11.5})
    span_b = InvocationStat(label="total", count=10, mean_ms=15.0, stddev_ms=1.0,
                           min_ms=13.0, max_ms=18.0, percentiles={0.95: 16.0})
    result_a = BenchmarkResult(label="baseline", metadata={}, total=span_a, operators=[])
    result_b = BenchmarkResult(label="candidate", metadata={}, total=span_b, operators=[])

    diff = result_a.diff(result_b)
    assert diff.total.mean_delta_ms == pytest.approx(5.0)
    assert diff.total.mean_delta_pct == pytest.approx(50.0)
    assert diff.total.percentile_deltas[0.95] == pytest.approx(4.5)


def test_diff_only_in_for_structural_difference():
    op_span = InvocationStat(label="0:op", count=10, mean_ms=5.0, stddev_ms=0.5,
                             min_ms=4.0, max_ms=6.0, percentiles={0.95: 5.8})
    total_span = InvocationStat(label="total", count=10, mean_ms=5.0, stddev_ms=0.5,
                                min_ms=4.0, max_ms=6.0, percentiles={0.95: 5.8})
    result_a = BenchmarkResult(label="a", metadata={}, total=total_span, operators=[op_span])
    result_b = BenchmarkResult(label="b", metadata={}, total=total_span, operators=[])

    diff = result_a.diff(result_b)
    assert len(diff.operators) == 1
    assert diff.operators[0].only_in == "baseline"
    assert diff.operators[0].mean_delta_ms is None


def test_diff_to_table_renders():
    result_a = _make_result("a")
    result_b = _make_result("b")
    diff = result_a.diff(result_b)
    table = diff.to_table()
    assert "baseline" in table
    assert "candidate" in table
    assert "Δmean" in table


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def test_benchmark_run_returns_result():
    config = MeasurementConfig(runs=5, warmup=2, percentiles=(0.50, 0.95))
    bench = Benchmark(
        pipeline=_make_pipeline(),
        input_fn=lambda: _static_input(1),
        measurement=config,
        label="smoke",
        metadata={"note": "test"},
    )
    result = bench.run()
    assert isinstance(result, BenchmarkResult)
    assert result.label == "smoke"
    assert result.metadata["note"] == "test"
    assert result.total.count == 5
    assert set(result.total.percentiles.keys()) == {0.50, 0.95}


def test_benchmark_restores_prior_tracing_config():
    pipeline = _make_pipeline()
    prior_collector = PrintCollector()
    pipeline.set_tracing(prior_collector)
    prior_config = pipeline._tracing_config

    config = MeasurementConfig(runs=5, warmup=2)
    Benchmark(pipeline, lambda: _static_input(1), measurement=config).run()

    assert pipeline._tracing_config is prior_config


def test_benchmark_restores_none_tracing_config():
    pipeline = _make_pipeline()
    assert pipeline._tracing_config is None

    Benchmark(pipeline, lambda: _static_input(1), measurement=MeasurementConfig(runs=3, warmup=1)).run()

    assert pipeline._tracing_config is None


def test_benchmark_per_operator_spans_present():
    config = MeasurementConfig(runs=10, warmup=2)
    result = Benchmark(_make_pipeline(), lambda: _static_input(1), measurement=config).run()
    assert len(result.operators) == 2
    labels = {s.label for s in result.operators}
    assert any("AddOne" in lbl or "0:" in lbl for lbl in labels)
    assert any("Double" in lbl or "1:" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# BenchmarkSweep
# ---------------------------------------------------------------------------

def _make_sweep_pipeline(**_) -> Pipeline[int, int]:
    return _make_pipeline()


def test_sweep_run_produces_correct_count():
    configs = [{"workers": 1}, {"workers": 2}]
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=configs,
        data_factory=DataFactory.from_callable(lambda idx: lambda: _static_input(idx)),
        data_configs=[{"idx": 0}, {"idx": 1}],
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    assert len(results) == 4  # 2 configs × 2 data configs


def test_sweep_result_labels_contain_input_and_config():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"mode": "fast"}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        data_configs=[{"name": "my_input"}],
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    assert len(results) == 1
    assert "my_input" in results[0].label
    assert "fast" in results[0].label


def test_sweep_result_labels_include_data_config():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{}],
        data_factory=DataFactory.from_callable(lambda idx: lambda: _static_input(idx)),
        data_configs=[{"idx": 0}, {"idx": 1}],
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    assert "idx:0" in results[0].label
    assert "idx:1" in results[1].label


def test_sweep_default_measurement():
    sweep = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
    )
    assert sweep.measurement is not None
    assert sweep.measurement.runs == 100


def test_sweep_requires_pipeline_factory():
    with pytest.raises(TypeError, match="BenchmarkSweep.factory expects a PipelineFactory"):
        BenchmarkSweep(
            factory=_make_sweep_pipeline,
            configs=[{}],
            data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        )


def test_sweep_requires_data_factory():
    with pytest.raises(TypeError, match="BenchmarkSweep.data_factory expects a DataFactory"):
        BenchmarkSweep(
            factory=PipelineFactory.from_callable(_make_sweep_pipeline),
            configs=[{}],
            data_factory=lambda **_: lambda: _static_input(0),
        )


def test_sweep_to_table_renders():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"a": 1}, {"a": 2}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    table = BenchmarkResult.to_comparison_table(results)
    assert "total" in table
    assert "mean" in table


def test_sweep_to_table_expand_regions_false():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"a": 1}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    table = BenchmarkResult.to_comparison_table(results, expand_regions=False)
    assert "total" in table


def test_sweep_to_table_empty():
    assert BenchmarkResult.to_comparison_table([]) == "(no results)"


def test_comparison_table_union_of_percentile_columns():
    # result_a has p50+p95, result_b has p50+p99 — table must show all three
    result_a = _make_result_with_percentiles("a", (0.50, 0.95))
    result_b = _make_result_with_percentiles("b", (0.50, 0.99))
    table = BenchmarkResult.to_comparison_table([result_a, result_b])
    assert "p50" in table
    assert "p95" in table
    assert "p99" in table


def test_comparison_table_missing_percentile_renders_dash():
    # result_a has no p99 — that cell must render as "-", not "0.00"
    result_a = _make_result_with_percentiles("a", (0.50, 0.95))
    result_b = _make_result_with_percentiles("b", (0.50, 0.99))
    table = BenchmarkResult.to_comparison_table([result_a, result_b])
    assert "0.00" not in table  # "0.00" would indicate missing value treated as zero
    assert "-" in table         # dash present for the missing cell


def _make_result_with_percentiles(label: str, percentiles: tuple[float, ...]) -> BenchmarkResult:
    config = MeasurementConfig(runs=5, warmup=1, percentiles=percentiles)
    collector = BenchmarkCollector(config)
    pipeline = _make_pipeline()
    pipeline.set_tracing(collector)
    for _ in range(6):
        pipeline(1)
    collector.stop()
    return collector.report(label=label)


def test_sweep_metadata_records_config_and_data_config():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"batch": 4}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
    ).run()
    meta = results[0].metadata
    assert meta["pipeline_config"] == {"batch": 4}
    assert "data_config" in meta


def test_sweep_label_prefix_single_result():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
        label_prefix="baseline",
    ).run()
    assert results[0].label == "baseline"


def test_sweep_label_prefix_multi_result():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"a": 1}, {"a": 2}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
        label_prefix="exp",
    ).run()
    assert len(results) == 2
    assert all(r.label.startswith("exp|") for r in results)


def test_sweep_extra_metadata_merged_into_all_results():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"a": 1}, {"a": 2}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
        extra_metadata={"env": "ci", "git_sha": "abc"},
    ).run()
    assert len(results) == 2
    for r in results:
        assert r.metadata["env"] == "ci"
        assert r.metadata["git_sha"] == "abc"


def test_sweep_extra_metadata_does_not_overwrite_pipeline_config():
    results = BenchmarkSweep(
        factory=PipelineFactory.from_callable(_make_sweep_pipeline),
        configs=[{"batch": 4}],
        data_factory=DataFactory.from_callable(lambda **_: lambda: _static_input(0)),
        measurement=MeasurementConfig(runs=3, warmup=1),
        extra_metadata={"note": "hi"},
    ).run()
    assert results[0].metadata["pipeline_config"] == {"batch": 4}
    assert results[0].metadata["note"] == "hi"


# ---------------------------------------------------------------------------
# BenchmarkBuilder — sweep expansion
# ---------------------------------------------------------------------------

def _fixed_input_factory(**_):
    return lambda: _static_input(0)


def test_builder_pipeline_configs_expand_cartesian_product():
    pipeline_configs = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1, 2)
        .pipeline_config_axis("b", "x", "y")
        .pipeline_config_axis("c", True)
        .data_factory(_fixed_input_factory)
        ._resolve_pipeline_configs()
    )
    assert len(pipeline_configs) == 4  # 2 × 2 × 1
    assert {"a": 1, "b": "x", "c": True} in pipeline_configs
    assert {"a": 2, "b": "y", "c": True} in pipeline_configs


def test_builder_sweep_run_produces_correct_count():
    results = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("mode", "fast", "slow")
        .data_factory(lambda idx: lambda: _static_input(idx))
        .data_config_axis("idx", 0, 1)
        .runs(3).warmup(1)
        .run()
    )
    assert len(results) == 8  # (2×2 pipeline) × 2 data configs


def test_builder_single_axis_sweep():
    results = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("conf", 0.1, 0.5, 0.9)
        .data_factory(_fixed_input_factory)
        .runs(3).warmup(1)
        .run()
    )
    assert len(results) == 3


def test_builder_sweep_default_measurement():
    builder = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1)
        .data_factory(_fixed_input_factory)
    )
    m = builder._build_measurement()
    assert m.runs == 100


def test_builder_sweep_filter_removes_invalid_combos():
    pipeline_configs = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2, 4)
        .pipeline_config_axis("batch_size", 1, 2, 4)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        ._resolve_pipeline_configs()
    )
    assert all(c["workers"] >= c["batch_size"] for c in pipeline_configs)
    assert len(pipeline_configs) == 6  # (1,1),(2,1),(2,2),(4,1),(4,2),(4,4)


def test_builder_plan_shows_all_combos():
    plan = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("batch_size", 1, 2)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        .plan()
    )
    assert "○" in plan
    assert "×" in plan
    assert "4 combinations" in plan
    assert "3 active" in plan
    assert "1 filtered" in plan


def test_builder_plan_without_filter_marks_all_active():
    plan = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1, 2)
        .pipeline_config_axis("b", 1, 2)
        .data_factory(_fixed_input_factory)
        .plan()
    )
    assert "4 active, 0 filtered" in plan
    assert "×" not in plan.split("grid:")[0]


def test_plan_includes_grid_for_2axes():
    plan = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("batch_size", 1, 2)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        .plan()
    )
    assert "row=workers" in plan
    assert "col=batch_size" in plan


def test_plan_includes_grid_for_3axes():
    plan = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("batch_size", 1, 2)
        .pipeline_config_axis("serialize", True, False)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        .plan()
    )
    assert "grp=serialize" in plan


def test_plan_single_axis_omits_grid():
    plan = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1, 2)
        .data_factory(_fixed_input_factory)
        .plan()
    )
    assert "grid:" not in plan
    assert "2 combinations" in plan


def test_grid_2axes():
    grid = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("batch_size", 1, 2)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        .grid()
    )
    assert "○" in grid
    assert "×" in grid
    assert "row=workers" in grid
    assert "col=batch_size" in grid


def test_grid_3axes():
    grid = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("workers", 1, 2)
        .pipeline_config_axis("batch_size", 1, 2)
        .pipeline_config_axis("serialize", True, False)
        .pipeline_config_filter(lambda c: c["workers"] >= c["batch_size"])
        .data_factory(_fixed_input_factory)
        .grid()
    )
    assert "grp=serialize" in grid


def test_grid_wrong_axes_raises():
    with pytest.raises(ValueError):
        (
            BenchmarkBuilder.factory(_make_sweep_pipeline)
            .pipeline_config_axis("a", 1)
            .data_factory(_fixed_input_factory)
            .grid()
        )


def test_builder_sweep_filter_none_keeps_all():
    pipeline_configs = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1, 2)
        .pipeline_config_axis("b", 1, 2)
        .data_factory(_fixed_input_factory)
        ._resolve_pipeline_configs()
    )
    assert len(pipeline_configs) == 4


def test_builder_sweep_to_comparison_table_renders():
    results = (
        BenchmarkBuilder.factory(_make_sweep_pipeline)
        .pipeline_config_axis("a", 1, 2)
        .data_factory(_fixed_input_factory)
        .runs(3).warmup(1)
        .run()
    )
    assert "total" in BenchmarkResult.to_comparison_table(results)


# ---------------------------------------------------------------------------
# BenchmarkResult.slug
# ---------------------------------------------------------------------------

def test_slug_no_illegal_chars():
    result = _make_result(label="coco.jpg|batch_size:4|serialize:True")
    slug = result.slug()
    for ch in r'/\:*?"<>|':
        assert ch not in slug


def test_slug_with_extension():
    result = _make_result(label="run_a")
    assert result.slug(".json") == "run_a.json"


def test_slug_colon_replaced():
    result = _make_result(label="img|batch_size:4")
    assert ":" not in result.slug()


# ---------------------------------------------------------------------------
# BenchmarkBuilder
# ---------------------------------------------------------------------------

from ml_pipes.factory import pipeline_factory as _pipeline_factory, data_factory as _data_factory

@_pipeline_factory
def _builder_pipeline_factory(**_) -> Pipeline[int, int]:
    return Pipeline([_AddOne(), _Double()])


def _builder_input_fn():
    return ("input", 1, None, None)


@_data_factory
def _builder_data_factory(value: int = 1):
    """Data factory: accepts a config dict, returns an InputFn."""
    def input_fn():
        return (str(value), 1, None, None)   # always pass int 1 to the pipeline
    return input_fn


# --- Named constructors ---

def test_builder_pipeline_constructor():
    p = _make_pipeline()
    b = BenchmarkBuilder.pipeline(p)
    assert b._pipeline_source is p


def test_builder_factory_constructor():
    b = BenchmarkBuilder.factory(_builder_pipeline_factory)
    assert b._pipeline_source is _builder_pipeline_factory


def test_builder_factory_constructor_preserves_plain_dict_factory():
    def make_pipeline(labels: dict[str, int]) -> Pipeline[int, int]:
        return _make_pipeline()

    b = BenchmarkBuilder.factory(make_pipeline)
    assert isinstance(b._pipeline_source, PipelineFactory)
    assert b._pipeline_source is not make_pipeline
    assert isinstance(b._pipeline_source.build({"labels": {"spam": 1}}), Pipeline)


def test_builder_factory_constructor_preserves_plain_keyword_factory():
    def make_pipeline(value: int = 1) -> Pipeline[int, int]:
        return _make_pipeline()

    b = BenchmarkBuilder.factory(make_pipeline)
    assert isinstance(b._pipeline_source, PipelineFactory)
    assert b._pipeline_source is not make_pipeline
    assert isinstance(b._pipeline_source.build({"value": 2}), Pipeline)


def test_builder_data_factory_assignment_coerces_decorated_factory():
    b = BenchmarkBuilder.pipeline(_make_pipeline()).data_factory(_builder_data_factory)
    assert b._data_factory is _builder_data_factory


def test_builder_data_factory_assignment_preserves_plain_dict_factory():
    def make_data(labels: dict[str, int]):
        return _builder_input_fn

    b = BenchmarkBuilder.pipeline(_make_pipeline()).data_factory(make_data)
    assert isinstance(b._data_factory, DataFactory)
    assert b._data_factory is not make_data
    assert b._data_factory.build({"labels": {"spam": 1}}) is _builder_input_fn


def test_builder_resolved_concrete_pipeline_rejects_config_keys():
    factory = BenchmarkBuilder.pipeline(_make_pipeline())._resolve_pipeline_factory()
    with pytest.raises(TypeError, match="unknown config key"):
        factory.validate_config({"workers": 4})


def test_builder_resolved_concrete_input_rejects_config_keys():
    factory = BenchmarkBuilder.pipeline(_make_pipeline()).data_input(_builder_input_fn)._resolve_data_factory()
    with pytest.raises(TypeError, match="unknown config key"):
        factory.validate_config({"value": 1})


def test_builder_data_inputs_resolve_by_label():
    def other_input_fn():
        return ("other", 2, None, None)

    builder = BenchmarkBuilder.pipeline(_make_pipeline()).data_inputs(
        [_builder_input_fn, other_input_fn],
        ["first", "second"],
    )
    factory = builder._resolve_data_factory()
    assert factory.build({"_label": "first"}) is _builder_input_fn
    assert factory.build({"_label": "second"}) is other_input_fn


def test_builder_data_inputs_reject_extra_config_keys():
    builder = BenchmarkBuilder.pipeline(_make_pipeline()).data_inputs(
        [_builder_input_fn],
        ["first"],
    )
    factory = builder._resolve_data_factory()
    with pytest.raises(TypeError, match="unknown config key"):
        factory.validate_config({"_label": "first", "value": 1})


def test_builder_data_inputs_require_matching_lengths():
    with pytest.raises(ValueError, match="same number of input functions and labels"):
        BenchmarkBuilder.pipeline(_make_pipeline()).data_inputs(
            [_builder_input_fn],
            ["first", "second"],
        )


def test_builder_data_inputs_require_unique_labels():
    with pytest.raises(ValueError, match="unique labels"):
        BenchmarkBuilder.pipeline(_make_pipeline()).data_inputs(
            [_builder_input_fn, _builder_input_fn],
            ["first", "first"],
        )


# --- Single run: concrete pipeline + concrete input ---

def test_builder_single_run_returns_one_result():
    results = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_input(_builder_input_fn)
        .runs(3).warmup(1)
        .run()
    )
    assert len(results) == 1
    assert isinstance(results[0], BenchmarkResult)


def test_builder_single_run_label_applied():
    results = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_input(_builder_input_fn)
        .label("my-run")
        .runs(3).warmup(1)
        .run()
    )
    assert results[0].label == "my-run"


def test_builder_label_applied_with_factory_single_result():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .data_factory(_builder_data_factory)
        .label("factory-run")
        .runs(3).warmup(1)
        .run()
    )
    assert len(results) == 1
    assert results[0].label == "factory-run"


def test_builder_label_prefix_on_sweep():
    # with multiple results, label_prefix is prepended to each auto-generated label
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_set([{"x": 1}, {"x": 2}])
        .data_input(_builder_input_fn)
        .label("exp")
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2
    assert all(r.label.startswith("exp|") for r in results)


def test_builder_metadata_applied_to_all_results():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_set([{"x": 1}, {"x": 2}])
        .data_input(_builder_input_fn)
        .metadata({"env": "test", "git_sha": "abc123"})
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2
    for r in results:
        assert r.metadata["env"] == "test"
        assert r.metadata["git_sha"] == "abc123"


def test_builder_metadata_does_not_overwrite_pipeline_config():
    # metadata merges on top; pipeline_config key in auto-metadata is preserved
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(x=1)
        .data_input(_builder_input_fn)
        .metadata({"note": "hi"})
        .runs(2).warmup(1)
        .run()
    )
    assert results[0].metadata["pipeline_config"] == {"x": 1}
    assert results[0].metadata["note"] == "hi"


def test_builder_label_and_metadata_apply_with_data_factory_sweep():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_set([{"x": 1}, {"x": 2}])
        .data_factory(_builder_data_factory)
        .data_config(value=7)
        .label("factory-exp")
        .metadata({"env": "test"})
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2
    assert all(r.label.startswith("factory-exp|") for r in results)
    for expected_x, result in zip((1, 2), results):
        assert result.metadata["pipeline_config"] == {"x": expected_x}
        assert result.metadata["data_config"] == {"value": 7}
        assert result.metadata["env"] == "test"


# --- Sweep: factory + explicit pipeline configs + concrete input ---

def test_builder_pipeline_configs_sweep():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_set([{}, {}])
        .data_input(_builder_input_fn)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2


def test_builder_pipeline_config_builds_dict():
    b = BenchmarkBuilder.factory(_builder_pipeline_factory)
    b.pipeline_config(workers=4)
    assert b._pipeline_config_dict == {"workers": 4}


# --- Sweep: concrete pipeline + data_factory + data configs ---

def test_builder_data_factory_sweep():
    results = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config_set([{"value": 1}, {"value": 2}])
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2


def test_builder_data_config_builds_dict():
    b = BenchmarkBuilder.pipeline(_make_pipeline()).data_factory(_builder_data_factory)
    b.data_config(value=99)
    assert b._data_config_dict == {"value": 99}


# --- Cross sweep: factory configs × data configs ---

def test_builder_pipeline_configs_cross_data_configs():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_set([{}, {}])
        .data_factory(_builder_data_factory)
        .data_config_set([{"value": 1}, {"value": 2}])
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 4  # 2 pipeline configs × 2 data configs


# --- Sweep axes: pipeline ---

def test_builder_pipeline_config_axis_expands_sweep():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_axis("x", 1, 2, 3)
        .data_input(_builder_input_fn)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 3


# --- Sweep axes: data ---

def test_builder_data_config_axis_expands_sweep():
    results = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config_axis("value", 10, 20)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2


# --- Sweep axes: pipeline × data ---

def test_builder_both_axes_cross_product():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_axis("x", 1, 2)
        .data_factory(_builder_data_factory)
        .data_config_axis("value", 10, 20)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 4  # 2 × 2


# --- plan() / grid() require axes ---

def test_builder_plan_without_axes_lists_configs():
    plan = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(x=1)
        .data_input(_builder_input_fn)
        .plan()
    )
    assert "1 config" in plan
    assert "x" in plan


def test_builder_plan_returns_string():
    b = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_axis("x", 1, 2, 3)
        .data_input(_builder_input_fn)
    )
    assert isinstance(b.plan(), str)


# --- Filters ---

def test_builder_pipeline_config_filter_drops_configs():
    results = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config_axis("x", 1, 2, 3, 4)
        .pipeline_config_filter(lambda c: c["x"] % 2 == 0)
        .data_input(_builder_input_fn)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2  # x=2 and x=4 kept


def test_builder_data_config_filter_drops_configs():
    results = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config_axis("value", 1, 2, 3, 4)
        .data_config_filter(lambda c: c["value"] % 2 == 0)
        .runs(2).warmup(1)
        .run()
    )
    assert len(results) == 2


# --- Validation errors ---

def test_builder_concrete_pipeline_with_pipeline_config_raises():
    with pytest.raises(ValueError, match="concrete Pipeline"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .pipeline_config(workers=4)
            .data_input(_builder_input_fn)
            .run()
        )


def test_builder_concrete_pipeline_with_pipeline_config_set_raises():
    with pytest.raises(ValueError, match="concrete Pipeline"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .pipeline_config_set([{"x": 1}])
            .data_input(_builder_input_fn)
            .run()
        )


def test_builder_concrete_pipeline_with_pipeline_config_axis_raises():
    with pytest.raises(ValueError, match="concrete Pipeline"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .pipeline_config_axis("x", 1, 2)
            .data_input(_builder_input_fn)
            .run()
        )


def test_builder_pipeline_configs_and_axis_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        (
            BenchmarkBuilder.factory(_builder_pipeline_factory)
            .pipeline_config_set([{}])
            .pipeline_config_axis("x", 1)
            .data_input(_builder_input_fn)
            .run()
        )


def test_builder_data_configs_and_axis_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .data_factory(_builder_data_factory)
            .data_config_set([{}])
            .data_config_axis("value", 1)
            .run()
        )


def test_builder_data_input_and_data_factory_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .data_input(_builder_input_fn)
            .data_factory(_builder_data_factory)
            .run()
        )


def test_builder_data_config_with_data_input_raises():
    with pytest.raises(ValueError, match="concrete InputFn"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .data_input(_builder_input_fn)
            .data_config(value=1)
            .run()
        )


def test_builder_data_config_without_factory_raises():
    with pytest.raises(ValueError, match="data_factory"):
        (
            BenchmarkBuilder.pipeline(_make_pipeline())
            .data_config(value=1)
            .run()
        )


def test_builder_no_data_raises():
    with pytest.raises(ValueError, match="data_input.*data_factory"):
        BenchmarkBuilder.pipeline(_make_pipeline()).runs(2).warmup(1).run()


# --- Measurement defaults ---

def test_builder_measurement_defaults():
    b = BenchmarkBuilder.pipeline(_make_pipeline())
    m = b._build_measurement()
    assert m.runs == 100
    assert m.warmup == 10
    assert m.percentiles == (0.50, 0.95, 0.99)


def test_builder_measurement_custom():
    b = BenchmarkBuilder.pipeline(_make_pipeline()).runs(20).warmup(3).percentiles(0.5, 0.99)
    m = b._build_measurement()
    assert m.runs == 20
    assert m.warmup == 3
    assert m.percentiles == (0.5, 0.99)


# ---------------------------------------------------------------------------
# BenchmarkBuilder — pipeline config merging
# ---------------------------------------------------------------------------

def test_pipeline_config_and_pipeline_config_merge_into_single_config():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx")
        .pipeline_config(conf_threshold=0.5, slice_wh=(320, 320))
        ._resolve_pipeline_configs()
    )
    assert configs == [{"model_path": "/model.onnx", "conf_threshold": 0.5, "slice_wh": (320, 320)}]


def test_pipeline_config_merges_into_config_set():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx")
        .pipeline_config_set([{"slice_wh": (320, 320)}, {"slice_wh": (480, 480)}])
        ._resolve_pipeline_configs()
    )
    assert configs == [
        {"model_path": "/model.onnx", "slice_wh": (320, 320)},
        {"model_path": "/model.onnx", "slice_wh": (480, 480)},
    ]


def test_pipeline_config_multi_merges_into_config_set():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx", conf_threshold=0.25)
        .pipeline_config_set([{"slice_wh": (320, 320)}, {"slice_wh": (480, 480)}])
        ._resolve_pipeline_configs()
    )
    assert configs == [
        {"model_path": "/model.onnx", "conf_threshold": 0.25, "slice_wh": (320, 320)},
        {"model_path": "/model.onnx", "conf_threshold": 0.25, "slice_wh": (480, 480)},
    ]


def test_pipeline_config_merges_into_axis_expansion():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx")
        .pipeline_config_axis("slice_wh", (320, 320), (480, 480))
        ._resolve_pipeline_configs()
    )
    assert configs == [
        {"model_path": "/model.onnx", "slice_wh": (320, 320)},
        {"model_path": "/model.onnx", "slice_wh": (480, 480)},
    ]


def test_pipeline_config_multi_merges_into_axis_expansion():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx", conf_threshold=0.25)
        .pipeline_config_axis("slice_wh", (320, 320), (480, 480))
        ._resolve_pipeline_configs()
    )
    assert configs == [
        {"model_path": "/model.onnx", "conf_threshold": 0.25, "slice_wh": (320, 320)},
        {"model_path": "/model.onnx", "conf_threshold": 0.25, "slice_wh": (480, 480)},
    ]


def test_pipeline_config_set_key_overrides_base():
    # per-config entry wins over base when the same key appears in both
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(conf_threshold=0.25)
        .pipeline_config_set([{"conf_threshold": 0.5}, {}])
        ._resolve_pipeline_configs()
    )
    assert configs[0]["conf_threshold"] == 0.5
    assert configs[1]["conf_threshold"] == 0.25


def test_pipeline_axis_value_overrides_base():
    # axis value wins over base when the same key appears in both
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(conf_threshold=0.25)
        .pipeline_config_axis("conf_threshold", 0.1, 0.9)
        ._resolve_pipeline_configs()
    )
    assert [c["conf_threshold"] for c in configs] == [0.1, 0.9]


def test_pipeline_config_multi_present_in_each_axis_config():
    configs = (
        BenchmarkBuilder.factory(_builder_pipeline_factory)
        .pipeline_config(model_path="/model.onnx", output_path="/out")
        .pipeline_config_axis("slice_wh", (240, 240), (320, 320), (480, 480))
        ._resolve_pipeline_configs()
    )
    assert len(configs) == 3
    assert all(c["model_path"] == "/model.onnx" for c in configs)
    assert all(c["output_path"] == "/out" for c in configs)
    assert [c["slice_wh"] for c in configs] == [(240, 240), (320, 320), (480, 480)]


# ---------------------------------------------------------------------------
# BenchmarkBuilder — data config merging
# ---------------------------------------------------------------------------

def test_data_config_and_data_config_merge_into_single_config():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco")
        .data_config(split="val")
        ._resolve_data_configs()
    )
    assert configs == [{"dataset": "coco", "split": "val"}]


def test_data_config_merges_into_data_config_set():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco")
        .data_config_set([{"image_path": "a.jpg"}, {"image_path": "b.jpg"}])
        ._resolve_data_configs()
    )
    assert configs == [
        {"dataset": "coco", "image_path": "a.jpg"},
        {"dataset": "coco", "image_path": "b.jpg"},
    ]


def test_data_config_multi_merges_into_data_config_set():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco", split="val")
        .data_config_set([{"image_path": "a.jpg"}, {"image_path": "b.jpg"}])
        ._resolve_data_configs()
    )
    assert configs == [
        {"dataset": "coco", "split": "val", "image_path": "a.jpg"},
        {"dataset": "coco", "split": "val", "image_path": "b.jpg"},
    ]


def test_data_config_merges_into_data_axis():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco")
        .data_config_axis("value", 1, 2)
        ._resolve_data_configs()
    )
    assert all(c["dataset"] == "coco" for c in configs)
    assert [c["value"] for c in configs] == [1, 2]


def test_data_config_multi_merges_into_data_axis():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco", split="val")
        .data_config_axis("value", 1, 2)
        ._resolve_data_configs()
    )
    assert all(c["dataset"] == "coco" for c in configs)
    assert all(c["split"] == "val" for c in configs)
    assert [c["value"] for c in configs] == [1, 2]


def test_data_config_set_key_overrides_base():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(split="val")
        .data_config_set([{"split": "test"}, {}])
        ._resolve_data_configs()
    )
    assert configs[0]["split"] == "test"
    assert configs[1]["split"] == "val"


def test_data_axis_value_overrides_base():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(value=99)
        .data_config_axis("value", 1, 2)
        ._resolve_data_configs()
    )
    assert [c["value"] for c in configs] == [1, 2]


def test_data_config_multi_present_in_each_axis_config():
    configs = (
        BenchmarkBuilder.pipeline(_make_pipeline())
        .data_factory(_builder_data_factory)
        .data_config(dataset="coco", split="val")
        .data_config_axis("value", 1, 2, 3)
        ._resolve_data_configs()
    )
    assert len(configs) == 3
    assert all(c["dataset"] == "coco" for c in configs)
    assert all(c["split"] == "val" for c in configs)
    assert [c["value"] for c in configs] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Factory signature error feedback
# ---------------------------------------------------------------------------

from ml_pipes.factory import (
    pipeline_factory,
    data_factory as data_factory_decorator,
)


@pipeline_factory
def _strict_pipeline_factory(workers: int) -> Pipeline[int, int]:
    return _make_pipeline()


@data_factory_decorator
def _strict_data_factory(value: int) -> InputFn:
    def fn():
        return ("input", value, None, None)
    return fn


def test_pipeline_factory_bad_config_raises_with_context():
    sweep = BenchmarkSweep(
        factory=_strict_pipeline_factory,
        configs=[{"unknown_param": 1}],
        data_factory=DataFactory.from_callable(lambda **_: _static_input),
    )
    with pytest.raises(TypeError, match="pipeline factory got unknown config key.*unknown_param"):
        sweep.run()


def test_pipeline_factory_missing_required_raises_with_context():
    sweep = BenchmarkSweep(
        factory=_strict_pipeline_factory,
        configs=[{}],
        data_factory=DataFactory.from_callable(lambda **_: _static_input),
    )
    with pytest.raises(TypeError, match="pipeline factory is missing required config key.*workers"):
        sweep.run()


def test_data_factory_bad_config_raises_with_context():
    sweep = BenchmarkSweep(
        factory=PipelineFactory.from_callable(lambda **_: _make_pipeline()),
        configs=[{}],
        data_factory=_strict_data_factory,
        data_configs=[{"unknown_param": 1}],
    )
    with pytest.raises(TypeError, match="data factory got unknown config key.*unknown_param"):
        sweep.run()


def test_data_factory_missing_required_raises_with_context():
    sweep = BenchmarkSweep(
        factory=PipelineFactory.from_callable(lambda **_: _make_pipeline()),
        configs=[{}],
        data_factory=_strict_data_factory,
        data_configs=[{}],
    )
    with pytest.raises(TypeError, match="data factory is missing required config key.*value"):
        sweep.run()


def test_pipeline_factory_returns_none_raises_with_context():
    sweep = BenchmarkSweep(
        factory=PipelineFactory.from_callable(lambda **_: None),
        configs=[{"workers": 1}],
        data_factory=DataFactory.from_callable(lambda **_: _static_input),
    )
    with pytest.raises(TypeError, match="pipeline factory must return a Pipeline"):
        sweep.run()


def test_data_factory_returns_none_raises_with_context():
    sweep = BenchmarkSweep(
        factory=PipelineFactory.from_callable(lambda **_: _make_pipeline()),
        configs=[{}],
        data_factory=DataFactory.from_callable(lambda **_: None),
    )
    with pytest.raises(TypeError, match="data factory must return a callable InputFn"):
        sweep.run()
