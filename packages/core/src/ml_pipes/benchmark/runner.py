from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..collectors.concurrent_collector import ConcurrentCollector
from ..core import Pipeline
from ..factory import DataFactory, InputFn, PipelineFactory
from ..tracing import InvocationTrace
from .results import BenchmarkResult, InvocationStat


@dataclass
class MeasurementConfig:
    runs: int = 100
    warmup: int = 10
    percentiles: tuple[float, ...] = (0.50, 0.95, 0.99)


@dataclass
class _SpanAccum:
    """Mutable accumulator node — mirrors the StepSpan tree during collection."""

    label: str
    samples: list[float] = field(default_factory=list)
    children: list["_SpanAccum"] = field(default_factory=list)


def _compute_stat(label: str, samples_s: list[float], percentiles: tuple[float, ...]) -> InvocationStat:
    arr = np.array(samples_s) * 1000.0
    return InvocationStat(
        label=label,
        count=len(arr),
        mean_ms=float(np.mean(arr)),
        stddev_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        percentiles={p: float(np.percentile(arr, p * 100)) for p in percentiles},
    )


class BenchmarkCollector(ConcurrentCollector):
    """Accumulates per-operator raw latency samples across runs, skipping warmup.

    Attach to a pipeline via set_tracing(), then call report() after all runs.
    Can be used standalone without Benchmark.
    """

    def __init__(self, config: MeasurementConfig) -> None:
        super().__init__()
        self._config = config
        self._calls = 0
        self._total_samples: list[float] = []
        self._span_tree: list[_SpanAccum] = []

    def _collect_spans(self, spans: list, accum_list: list[_SpanAccum]) -> None:
        accum_by_label: dict[str, _SpanAccum] = {accum.label: accum for accum in accum_list}
        for span in spans:
            if span.label not in accum_by_label:
                node = _SpanAccum(span.label)
                accum_list.append(node)
                accum_by_label[span.label] = node
            node = accum_by_label[span.label]
            node.samples.append(span.duration_s)
            if span.child_trace is not None:
                self._collect_spans(span.child_trace.spans, node.children)

    def _collect(self, trace: InvocationTrace) -> None:
        self._calls += 1
        if self._calls <= self._config.warmup:
            return
        self._total_samples.append(trace.total_duration_s)
        self._collect_spans(trace.spans, self._span_tree)

    def report(self, label: str = "", metadata: dict | None = None) -> BenchmarkResult:
        self.flush()
        pct = self._config.percentiles

        def _build(accum_list: list[_SpanAccum]) -> list[InvocationStat]:
            stats = []
            for node in accum_list:
                stat = _compute_stat(node.label, node.samples, pct)
                stat.children = _build(node.children)
                stats.append(stat)
            return stats

        total = (
            _compute_stat("total", self._total_samples, pct)
            if self._total_samples
            else _compute_stat("total", [0.0], pct)
        )
        operators = _build(self._span_tree)
        return BenchmarkResult(
            label=label,
            metadata=metadata or {},
            total=total,
            operators=operators,
        )

    def reset(self) -> None:
        self.flush()
        self._calls = 0
        self._total_samples.clear()
        self._span_tree.clear()


class Benchmark:
    """Drives a measurement loop: warmup + N measured runs → BenchmarkResult.

    input_fn must return (id, value, tag, metadata).
    tag and metadata are accepted but currently ignored; reserved for future features.

    Saves and restores the pipeline's prior tracing config after run().
    """

    def __init__(
        self,
        pipeline: Pipeline[Any, Any],
        input_fn: InputFn,
        measurement: MeasurementConfig = MeasurementConfig(),
        label: str = "",
        metadata: dict | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._input_fn = input_fn
        self._measurement = measurement
        self._label = label
        self._metadata = metadata or {}

    def run(self) -> BenchmarkResult:
        prior_tracing = self._pipeline._tracing_config
        collector = BenchmarkCollector(self._measurement)
        self._pipeline.set_tracing(collector)
        try:
            total_runs = self._measurement.warmup + self._measurement.runs
            for _ in range(total_runs):
                _id, value, _tag, _meta = self._input_fn()
                self._pipeline(value)
        finally:
            collector.stop()
            self._pipeline._tracing_config = prior_tracing

        return collector.report(label=self._label, metadata=self._metadata)


def _format_config_label(config: dict, *, default: str) -> str:
    if "_label" in config and len(config) == 1:
        return str(config["_label"])

    visible = {key: value for key, value in config.items() if not key.startswith("_")}
    return "|".join(f"{key}:{value}" for key, value in visible.items()) or default


@dataclass
class BenchmarkSweep:
    """Cross-product every pipeline config with every data config and collect all results.

    Pass the values returned by ``@pipeline_factory`` and ``@data_factory``.
    That is the normal way to use ``BenchmarkSweep``.

    ``data_factory`` is called fresh for each data config::

        sweep = BenchmarkSweep(
            factory=make_pipeline,
            configs=[{"workers": 1}, {"workers": 4}],
            data_factory=make_input,
            data_configs=[{"image": "a.jpg"}, {"image": "b.jpg"}],
            measurement=MeasurementConfig(runs=30),
        )

    If you need to adapt a plain callable explicitly, wrap it with
    ``PipelineFactory.from_callable(...)`` or ``DataFactory.from_callable(...)``.
    """

    factory: PipelineFactory[Any, Any]
    configs: list[dict]
    data_factory: DataFactory
    data_configs: list[dict] | None = None
    measurement: MeasurementConfig = None  # type: ignore[assignment]
    label_prefix: str | None = None
    extra_metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.measurement is None:
            self.measurement = MeasurementConfig()
        if self.data_configs is None:
            self.data_configs = [{}]
        if not isinstance(self.factory, PipelineFactory):
            raise TypeError(
                "BenchmarkSweep.factory expects a PipelineFactory. "
                "Pass @pipeline_factory output or PipelineFactory.from_callable(...)."
            )
        if not isinstance(self.data_factory, DataFactory):
            raise TypeError(
                "BenchmarkSweep.data_factory expects a DataFactory. "
                "Pass @data_factory output or DataFactory.from_callable(...)."
            )

    def run(self) -> list[BenchmarkResult]:
        data_configs = self.data_configs if self.data_configs is not None else [{}]
        is_single = len(self.configs) == 1 and len(data_configs) == 1
        results: list[BenchmarkResult] = []
        for data_config in data_configs:
            input_fn = self.data_factory.build(data_config)
            data_label = _format_config_label(data_config, default="input")
            for pipeline_config in self.configs:
                pipeline_label = _format_config_label(pipeline_config, default="")
                auto_label = "|".join(part for part in (data_label, pipeline_label) if part)
                if self.label_prefix:
                    label = self.label_prefix if is_single else f"{self.label_prefix}|{auto_label}"
                else:
                    label = auto_label
                metadata: dict = {"pipeline_config": pipeline_config, "data_config": data_config}
                if self.extra_metadata:
                    metadata.update(self.extra_metadata)
                pipeline = self.factory.build(pipeline_config)
                result = Benchmark(
                    pipeline=pipeline,
                    input_fn=input_fn,
                    measurement=self.measurement,
                    label=label,
                    metadata=metadata,
                ).run()
                results.append(result)
        return results
