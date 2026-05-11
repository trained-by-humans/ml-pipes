from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ml_pipes import Pipeline
from ml_pipes.benchmark import (
    Benchmark,
    BenchmarkCollector,
    BenchmarkMatrix,
    BenchmarkSweep,
    BenchmarkResult,
    InvocationStat,
    InvocationStatDiff,
    MeasurementConfig,
)
from ml_pipes.tracing import InvocationTrace, StepSpan, TracingConfig
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


def _make_pipeline() -> Pipeline:
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
        config=config,
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
    Benchmark(pipeline, lambda: _static_input(1), config=config).run()

    assert pipeline._tracing_config is prior_config


def test_benchmark_restores_none_tracing_config():
    pipeline = _make_pipeline()
    assert pipeline._tracing_config is None

    Benchmark(pipeline, lambda: _static_input(1), config=MeasurementConfig(runs=3, warmup=1)).run()

    assert pipeline._tracing_config is None


def test_benchmark_per_operator_spans_present():
    config = MeasurementConfig(runs=10, warmup=2)
    result = Benchmark(_make_pipeline(), lambda: _static_input(1), config=config).run()
    assert len(result.operators) == 2
    labels = {s.label for s in result.operators}
    assert any("AddOne" in lbl or "0:" in lbl for lbl in labels)
    assert any("Double" in lbl or "1:" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# BenchmarkSweep
# ---------------------------------------------------------------------------

def _make_pipeline_from_config(config: dict) -> Pipeline:
    return _make_pipeline()


def test_matrix_run_produces_correct_count():
    configs = [{"workers": 1}, {"workers": 2}]
    inputs = [lambda: _static_input(0), lambda: _static_input(1)]
    matrix = BenchmarkSweep(
        pipeline_factory=_make_pipeline_from_config,
        pipeline_configs=configs,
        inputs=inputs,
        config=MeasurementConfig(runs=3, warmup=1),
    )
    results = matrix.run()
    assert len(results) == 4  # 2 configs × 2 inputs


def test_matrix_result_labels_contain_input_and_config():
    configs = [{"mode": "fast"}]
    inputs = [lambda: ("my_input", 0, None, None)]
    matrix = BenchmarkSweep(
        pipeline_factory=_make_pipeline_from_config,
        pipeline_configs=configs,
        inputs=inputs,
        config=MeasurementConfig(runs=3, warmup=1),
    )
    results = matrix.run()
    assert len(results) == 1
    assert "my_input" in results[0].label
    assert "fast" in results[0].label


def test_matrix_default_config():
    matrix = BenchmarkSweep(
        pipeline_factory=_make_pipeline_from_config,
        pipeline_configs=[{}],
        inputs=[lambda: _static_input(0)],
    )
    assert matrix.config is not None
    assert matrix.config.runs == 100


def test_matrix_to_table_renders():
    configs = [{"a": 1}, {"a": 2}]
    inputs = [lambda: _static_input(0)]
    results = BenchmarkSweep(
        pipeline_factory=_make_pipeline_from_config,
        pipeline_configs=configs,
        inputs=inputs,
        config=MeasurementConfig(runs=3, warmup=1),
    ).run()
    table = BenchmarkSweep.to_table(results)
    assert "total" in table
    assert "mean" in table


def test_matrix_to_table_empty():
    assert BenchmarkSweep.to_table([]) == "(no results)"


def test_matrix_metadata_records_config_and_input():
    configs = [{"batch": 4}]
    inputs = [lambda: ("img.jpg", 0, None, None)]
    results = BenchmarkSweep(
        pipeline_factory=_make_pipeline_from_config,
        pipeline_configs=configs,
        inputs=inputs,
        config=MeasurementConfig(runs=3, warmup=1),
    ).run()
    meta = results[0].metadata
    assert meta["pipeline_config"] == {"batch": 4}
    assert meta["input"] == "img.jpg"


# ---------------------------------------------------------------------------
# BenchmarkMatrix
# ---------------------------------------------------------------------------

def test_matrix_pipeline_configs_cartesian_product():
    matrix = BenchmarkMatrix(
        pipeline_factory=_make_pipeline_from_config,
        axes={"a": [1, 2], "b": ["x", "y"], "c": [True]},
        inputs=[lambda: _static_input(0)],
    )
    configs = matrix.pipeline_configs()
    assert len(configs) == 4  # 2 × 2 × 1
    assert {"a": 1, "b": "x", "c": True} in configs
    assert {"a": 2, "b": "y", "c": True} in configs


def test_matrix_run_produces_correct_count():
    matrix = BenchmarkMatrix(
        pipeline_factory=_make_pipeline_from_config,
        axes={"workers": [1, 2], "mode": ["fast", "slow"]},
        inputs=[lambda: _static_input(0), lambda: _static_input(1)],
        config=MeasurementConfig(runs=3, warmup=1),
    )
    results = matrix.run()
    assert len(results) == 8  # 2 axes × 2 values each × 2 inputs


def test_matrix_single_axis():
    matrix = BenchmarkMatrix(
        pipeline_factory=_make_pipeline_from_config,
        axes={"conf": [0.1, 0.5, 0.9]},
        inputs=[lambda: _static_input(0)],
        config=MeasurementConfig(runs=3, warmup=1),
    )
    results = matrix.run()
    assert len(results) == 3


def test_matrix_default_config():
    matrix = BenchmarkMatrix(
        pipeline_factory=_make_pipeline_from_config,
        axes={"a": [1]},
        inputs=[lambda: _static_input(0)],
    )
    assert matrix.config is not None
    assert matrix.config.runs == 100


def test_matrix_to_table_delegates_to_sweep():
    results = BenchmarkMatrix(
        pipeline_factory=_make_pipeline_from_config,
        axes={"a": [1, 2]},
        inputs=[lambda: _static_input(0)],
        config=MeasurementConfig(runs=3, warmup=1),
    ).run()
    assert BenchmarkMatrix.to_table(results) == BenchmarkSweep.to_table(results)
