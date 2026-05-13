from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

from .factory import InputFn

import numpy as np

from .collectors.concurrent_collector import ConcurrentCollector
from .core import Pipeline
from .tracing import InvocationTrace

PipelineFactory: TypeAlias = Callable[[dict], Pipeline]
DataFactory: TypeAlias = Callable[[dict], InputFn]
ConfigFilter: TypeAlias = Callable[[dict], bool]


@dataclass
class InvocationStat:
    """Latency statistics for one operator or the whole pipeline across N runs.

    Children mirror StepSpan.child_trace: present when the operator is a region
    (e.g. Scatter) and child spans were collected.
    """

    label: str
    count: int
    mean_ms: float
    stddev_ms: float        # run-to-run jitter; high value → noisy operator
    min_ms: float
    max_ms: float
    percentiles: dict[float, float]  # keys mirror MeasurementConfig.percentiles exactly
    children: list[InvocationStat] = field(default_factory=list)


def _flat_stats(
    stats: list[InvocationStat], expand_regions: bool = True, depth: int = 0
) -> list[tuple[int, InvocationStat]]:
    rows: list[tuple[int, InvocationStat]] = []
    for s in stats:
        rows.append((depth, s))
        if expand_regions and s.children:
            rows.extend(_flat_stats(s.children, expand_regions, depth + 1))
    return rows


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


@dataclass
class BenchmarkResult:
    """Portable benchmark artifact. `operators` is the ml-pipes-specific per-operator breakdown."""

    label: str
    metadata: dict                   # user-managed: env, device, git sha, config, dataset, etc.
    total: InvocationStat             # aggregate pipeline latency
    operators: list[InvocationStat]   # per-operator latency; each may have .children for region spans

    def to_table(self, expand_regions: bool = True) -> str:
        pct_keys = sorted(self.total.percentiles)
        pct_headers = [f"p{int(p * 100)}" for p in pct_keys]

        flat = [(0, self.total)] + _flat_stats(self.operators, expand_regions)

        col_label = max(len("  " * d + s.label) for d, s in flat)
        col_w = 9

        header = (
            f"{'operator':<{col_label}}   {'mean':>{col_w}}"
            + "".join(f"  {h:>{col_w}}" for h in pct_headers)
            + f"  {'stddev':>{col_w}}  {'min':>{col_w}}  {'max':>{col_w}}"
        )
        sep = "-" * len(header)

        lines = [header, sep]
        any_collapsed = False
        for depth, s in flat:
            indent = "  " * depth
            collapsed = not expand_regions and bool(s.children)
            if collapsed:
                any_collapsed = True
            label = indent + s.label + ("*" if collapsed else "")
            line = (
                f"{label:<{col_label}}  {s.mean_ms:>{col_w}.2f}"
                + "".join(f"  {s.percentiles[p]:>{col_w}.2f}" for p in pct_keys)
                + f"  {s.stddev_ms:>{col_w}.2f}  {s.min_ms:>{col_w}.2f}  {s.max_ms:>{col_w}.2f}"
            )
            lines.append(line)
        lines.append(sep)
        footer = f"runs: {self.total.count}  (all values in ms)"
        if any_collapsed:
            footer += "\n* Child spans are collapsed"
        lines.append(footer)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def _stat(s: InvocationStat) -> dict:
            d: dict = {
                "label": s.label,
                "count": s.count,
                "mean_ms": s.mean_ms,
                "stddev_ms": s.stddev_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
                "percentiles": {str(k): v for k, v in s.percentiles.items()},
            }
            if s.children:
                d["children"] = [_stat(c) for c in s.children]
            return d

        return {
            "label": self.label,
            "metadata": self.metadata,
            "total": _stat(self.total),
            "operators": [_stat(s) for s in self.operators],
        }

    def slug(self, ext: str = "") -> str:
        safe = re.sub(r'[/\\:*?"<>|]', "_", self.label)
        return safe + ext

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> BenchmarkResult:
        with open(path) as f:
            d = json.load(f)

        def _stat(s: dict) -> InvocationStat:
            return InvocationStat(
                label=s["label"],
                count=s["count"],
                mean_ms=s["mean_ms"],
                stddev_ms=s["stddev_ms"],
                min_ms=s["min_ms"],
                max_ms=s["max_ms"],
                percentiles={float(k): v for k, v in s["percentiles"].items()},
                children=[_stat(c) for c in s.get("children", [])],
            )

        return cls(
            label=d["label"],
            metadata=d.get("metadata", {}),
            total=_stat(d["total"]),
            operators=[_stat(s) for s in d["operators"]],
        )

    def diff(self, other: BenchmarkResult) -> BenchmarkDiff:
        return _make_diff(self, other)

    @staticmethod
    def to_comparison_table(results: list[BenchmarkResult], expand_regions: bool = True) -> str:
        """Render a multi-column comparison table: one column per result, rows are operators."""
        if not results:
            return "(no results)"

        all_rows: list[tuple[int, str]] = [(0, "total")]
        seen: set[str] = {"total"}
        for r in results:
            for depth, s in _flat_stats(r.operators, expand_regions):
                if s.label not in seen:
                    all_rows.append((depth, s.label))
                    seen.add(s.label)

        pct_keys = sorted(results[0].total.percentiles)

        col_label = max(len("  " * d + lbl) for d, lbl in all_rows)
        col_w = 9
        cols_per_result = 1 + len(pct_keys)
        result_col_w = cols_per_result * (col_w + 2)

        def _header_for(r: BenchmarkResult) -> str:
            label = r.label
            w = result_col_w - 1
            if len(label) > w:
                if "|" in label:
                    input_part, config_part = label.split("|", 1)
                    if len(config_part) <= w - 2:
                        keep = w - len(config_part) - 1
                        label = input_part[:keep] + "…|" + config_part
                    else:
                        label = config_part[-(w):]
                else:
                    half = (w - 1) // 2
                    label = label[:half] + "…" + label[len(label) - (w - half - 1):]
            return f"{label:<{result_col_w}}"

        def _subheader() -> str:
            sub = f"{'mean':>{col_w}}"
            for p in pct_keys:
                sub += f"  {f'p{int(p * 100)}':>{col_w}}"
            return sub

        def _row_for(stat: InvocationStat | None) -> str:
            if stat is None:
                return " " * col_w + "  " + "  ".join("-" * col_w for _ in pct_keys)
            row = f"{stat.mean_ms:>{col_w}.2f}"
            for p in pct_keys:
                row += f"  {stat.percentiles.get(p, 0.0):>{col_w}.2f}"
            return row

        lookups: list[dict[str, InvocationStat]] = []
        for r in results:
            d: dict[str, InvocationStat] = {"total": r.total}
            d.update({s.label: s for _, s in _flat_stats(r.operators, expand_regions=True)})
            lookups.append(d)

        sep_width = col_label + 3 + len(results) * (result_col_w + 2)
        sep = "-" * sep_width

        lines = [sep]
        lines.append(" " * (col_label + 3) + "  ".join(_header_for(r) for r in results))
        lines.append(" " * (col_label + 3) + "  ".join(_subheader() for _ in results))
        lines.append(sep)

        any_collapsed = False
        for depth, lbl in all_rows:
            indent = "  " * depth
            collapsed = not expand_regions and any(
                (s := lookup.get(lbl)) is not None and bool(s.children)
                for lookup in lookups
            )
            if collapsed:
                any_collapsed = True
            display = indent + lbl + ("*" if collapsed else "")
            row = f"{display:<{col_label}}  "
            row += "  ".join(_row_for(lookup.get(lbl)) for lookup in lookups)
            lines.append(row)

        lines.append(sep)
        footer = f"runs: {results[0].total.count}  (all values in ms)"
        if any_collapsed:
            footer += "\n* Child spans are collapsed"
        lines.append(footer)
        return "\n".join(lines)


@dataclass(frozen=True)
class InvocationStatDiff:
    label: str
    only_in: str | None                       # "baseline" | "candidate" | None (present in both)
    mean_delta_ms: float | None
    mean_delta_pct: float | None
    percentile_deltas: dict[float, float] | None  # ms delta per requested percentile, e.g. {0.95: +3.2}


@dataclass(frozen=True)
class BenchmarkDiff:
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    total: InvocationStatDiff
    operators: list[InvocationStatDiff]  # matched by label; unmatched have only_in set

    def to_table(self) -> str:
        rows = [self.total, *self.operators]
        all_pct_keys: list[float] = []
        for r in rows:
            if r.percentile_deltas:
                for k in r.percentile_deltas:
                    if k not in all_pct_keys:
                        all_pct_keys.append(k)
        all_pct_keys.sort()
        pct_headers = [f"Δp{int(p * 100)}" for p in all_pct_keys]

        col_label = max(len(r.label) for r in rows)
        col_w = 12

        header = (
            f"{'operator':<{col_label}}  {'Δmean':>{col_w}}  {'Δmean%':>{col_w}}"
            + "".join(f"  {h:>{col_w}}" for h in pct_headers)
            + f"  {'note':>10}"
        )
        sep = "-" * len(header)

        def _fmt(v: float | None, suffix: str = "") -> str:
            if v is None:
                return "-"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.2f}{suffix}"

        lines = [
            f"baseline : {self.baseline.label}",
            f"candidate: {self.candidate.label}",
            sep,
            header,
            sep,
        ]
        for r in rows:
            note = f"only in {r.only_in}" if r.only_in else ""
            pct_cols = "".join(
                f"  {_fmt(r.percentile_deltas.get(p) if r.percentile_deltas else None, 'ms'):>{col_w}}"
                for p in all_pct_keys
            )
            line = (
                f"{r.label:<{col_label}}"
                f"  {_fmt(r.mean_delta_ms, 'ms'):>{col_w}}"
                f"  {_fmt(r.mean_delta_pct, '%'):>{col_w}}"
                + pct_cols
                + f"  {note:>10}"
            )
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)


def _span_diff(label: str, b: InvocationStat | None, c: InvocationStat | None) -> InvocationStatDiff:
    if b is None:
        return InvocationStatDiff(label=label, only_in="candidate", mean_delta_ms=None, mean_delta_pct=None, percentile_deltas=None)
    if c is None:
        return InvocationStatDiff(label=label, only_in="baseline", mean_delta_ms=None, mean_delta_pct=None, percentile_deltas=None)

    mean_delta = c.mean_ms - b.mean_ms
    mean_pct = (mean_delta / b.mean_ms * 100) if b.mean_ms != 0 else None

    shared_pct = set(b.percentiles) & set(c.percentiles)
    pct_deltas = {p: c.percentiles[p] - b.percentiles[p] for p in sorted(shared_pct)}

    return InvocationStatDiff(
        label=label,
        only_in=None,
        mean_delta_ms=mean_delta,
        mean_delta_pct=mean_pct,
        percentile_deltas=pct_deltas if pct_deltas else None,
    )


def _make_diff(baseline: BenchmarkResult, candidate: BenchmarkResult) -> BenchmarkDiff:
    total_diff = _span_diff("total", baseline.total, candidate.total)

    b_by_label = {s.label: s for s in baseline.operators}
    c_by_label = {s.label: s for s in candidate.operators}
    all_labels: list[str] = []
    seen: set[str] = set()
    for s in baseline.operators:
        all_labels.append(s.label)
        seen.add(s.label)
    for s in candidate.operators:
        if s.label not in seen:
            all_labels.append(s.label)

    span_diffs = [_span_diff(lbl, b_by_label.get(lbl), c_by_label.get(lbl)) for lbl in all_labels]
    return BenchmarkDiff(baseline=baseline, candidate=candidate, total=total_diff, operators=span_diffs)


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
    children: list[_SpanAccum] = field(default_factory=list)


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
        accum_by_label: dict[str, _SpanAccum] = {a.label: a for a in accum_list}
        for span in spans:
            if span.label not in accum_by_label:
                node: _SpanAccum = _SpanAccum(span.label)
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

        total = _compute_stat("total", self._total_samples, pct) if self._total_samples else _compute_stat("total", [0.0], pct)
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
        pipeline: Pipeline,
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


@dataclass
class BenchmarkSweep:
    """Run pipeline_configs × data_configs and collect all results.

    The sweep cross-products every pipeline config with every data config,
    calling ``data_factory(data_config)`` fresh for each cell::

        sweep = BenchmarkSweep(
            factory=make_pipeline,
            configs=[{"workers": 1}, {"workers": 4}],
            data_factory=make_input,
            data_configs=[{"image": "a.jpg"}, {"image": "b.jpg"}],
            measurement=MeasurementConfig(runs=30),
        )

    To benchmark against a single fixed input with no config, wrap it::

        sweep = BenchmarkSweep(
            factory=make_pipeline,
            configs=[{"workers": 1}, {"workers": 4}],
            data_factory=lambda _: my_input_fn,
        )
    """

    factory: PipelineFactory
    configs: list[dict]
    data_factory: DataFactory
    data_configs: list[dict] | None = None
    measurement: MeasurementConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.measurement is None:
            self.measurement = MeasurementConfig()
        if self.data_configs is None:
            self.data_configs = [{}]

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for data_config in self.data_configs:  # type: ignore[union-attr]
            input_fn = self.data_factory(data_config)
            if "_label" in data_config and len(data_config) == 1:
                data_label = data_config["_label"]
            else:
                visible = {k: v for k, v in data_config.items() if not k.startswith("_")}
                data_label = "|".join(f"{k}:{v}" for k, v in visible.items()) or "input"
            for pipeline_config in self.configs:
                config_str = "|".join(f"{k}:{v}" for k, v in pipeline_config.items())
                label = f"{data_label}|{config_str}" if config_str else data_label
                result = Benchmark(
                    pipeline=self.factory(pipeline_config),
                    input_fn=input_fn,
                    measurement=self.measurement,
                    label=label,
                    metadata={"pipeline_config": pipeline_config, "data_config": data_config},
                ).run()
                results.append(result)
        return results



class BenchmarkBuilder:
    """Fluent builder for benchmarks.

    Entry points::

        BenchmarkBuilder.pipeline(p)   # concrete Pipeline (no config sweep)
        BenchmarkBuilder.factory(f)    # factory callable (config sweep / matrix)

    Chain measurement, pipeline config, and data config methods, then call
    ``.run()`` which returns ``list[BenchmarkResult]``.
    """

    def __init__(self, source: Pipeline | PipelineFactory) -> None:
        self._source = source

        self._data_input_fn: InputFn | None = None
        self._data_factory_fn: DataFactory | None = None

        self._pipeline_config_dict: dict = {}
        self._pipeline_configs_list: list[dict] | None = None
        self._pipeline_axes: dict[str, list] = {}
        self._pipeline_config_filter_fn: Callable | None = None

        self._data_config_dict: dict = {}
        self._data_configs_list: list[dict] | None = None
        self._data_axes: dict[str, list] = {}
        self._data_config_filter_fn: Callable | None = None

        self._runs: int | None = None
        self._warmup: int | None = None
        self._percentiles: list[float] | None = None
        self._label_str: str | None = None
        self._metadata_dict: dict | None = None

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def pipeline(cls, p: Pipeline) -> BenchmarkBuilder:
        """Start from a concrete Pipeline (no config sweep)."""
        return cls(p)

    @classmethod
    def factory(cls, f: PipelineFactory) -> BenchmarkBuilder:
        """Start from a pipeline factory callable."""
        return cls(f)

    # ------------------------------------------------------------------
    # Pipeline config dimension
    # ------------------------------------------------------------------

    def pipeline_config(self, **kwargs) -> BenchmarkBuilder:
        """Set a single pipeline config dict."""
        self._pipeline_config_dict.update(kwargs)
        return self

    def pipeline_configs(self, configs: list[dict]) -> BenchmarkBuilder:
        """Set an explicit list of pipeline configs for a sweep."""
        self._pipeline_configs_list = list(configs)
        return self

    def pipeline_config_arg(self, key: str, value: Any) -> BenchmarkBuilder:
        """Add a single key to the pipeline config dict."""
        self._pipeline_config_dict[key] = value
        return self

    def pipeline_config_axis(self, key: str, *values) -> BenchmarkBuilder:
        """Register a pipeline config axis for cartesian expansion."""
        self._pipeline_axes[key] = list(values)
        return self

    def pipeline_config_filter(self, pred: ConfigFilter) -> BenchmarkBuilder:
        """Drop pipeline configs where pred returns False."""
        self._pipeline_config_filter_fn = pred
        return self

    # ------------------------------------------------------------------
    # Data dimension
    # ------------------------------------------------------------------

    def data_input(self, fn: InputFn) -> BenchmarkBuilder:
        """Use a concrete InputFn (no data config sweep)."""
        self._data_input_fn = fn
        return self

    def data_inputs(self, fns: list[InputFn], labels: list[str]) -> BenchmarkBuilder:
        """Use multiple InputFns as a sweep — each gets a label."""
        _fn_map = dict(zip(labels, fns))
        self._data_factory_fn = lambda cfg, _m=_fn_map: _m[cfg["_label"]]
        self._data_configs_list = [{"_label": lab} for lab in labels]
        return self

    def data_factory(self, factory: DataFactory) -> BenchmarkBuilder:
        """Use a data factory callable (enables data config sweep)."""
        self._data_factory_fn = factory
        return self

    def data_config(self, **kwargs) -> BenchmarkBuilder:
        """Set a single data config dict."""
        self._data_config_dict.update(kwargs)
        return self

    def data_configs(self, configs: list[dict]) -> BenchmarkBuilder:
        """Set an explicit list of data configs for a sweep."""
        self._data_configs_list = list(configs)
        return self

    def data_config_arg(self, key: str, value: Any) -> BenchmarkBuilder:
        """Add a single key to the data config dict."""
        self._data_config_dict[key] = value
        return self

    def data_config_axis(self, key: str, *values) -> BenchmarkBuilder:
        """Register a data config axis for cartesian expansion."""
        self._data_axes[key] = list(values)
        return self

    def data_config_filter(self, pred: ConfigFilter) -> BenchmarkBuilder:
        """Drop data configs where pred returns False."""
        self._data_config_filter_fn = pred
        return self

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def runs(self, n: int) -> BenchmarkBuilder:
        self._runs = n
        return self

    def warmup(self, n: int) -> BenchmarkBuilder:
        self._warmup = n
        return self

    def percentiles(self, *ps: float) -> BenchmarkBuilder:
        self._percentiles = list(ps)
        return self

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def label(self, s: str) -> BenchmarkBuilder:
        self._label_str = s
        return self

    def metadata(self, d: dict) -> BenchmarkBuilder:
        self._metadata_dict = d
        return self

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _build_measurement(self) -> MeasurementConfig:
        n = self._runs or 100
        w = self._warmup if self._warmup is not None else max(5, n // 10)
        p = tuple(self._percentiles) if self._percentiles else (0.50, 0.95, 0.99)
        return MeasurementConfig(runs=n, warmup=w, percentiles=p)

    def _validate(self) -> None:
        if self._pipeline_configs_list is not None and self._pipeline_axes:
            raise ValueError("pipeline_configs() and pipeline_config_axis() are mutually exclusive")
        if self._data_configs_list is not None and self._data_axes:
            raise ValueError("data_configs() and data_config_axis() are mutually exclusive")
        if self._data_input_fn is not None and self._data_factory_fn is not None:
            raise ValueError("data_input() and data_factory() are mutually exclusive")
        has_data_config = (
            self._data_config_dict
            or self._data_configs_list is not None
            or self._data_axes
        )
        if has_data_config and self._data_input_fn is not None:
            raise ValueError("data_config*() cannot be used with data_input() — InputFn has no config")
        if has_data_config and self._data_factory_fn is None and self._data_input_fn is None:
            raise ValueError("data_config*() requires data_factory() to be set")

    def plan(self) -> str:
        output = self._plan()
        print(output, file=sys.stderr)
        return output

    def _plan(self) -> str:
        if not self._pipeline_axes:
            configs = self._resolve_pipeline_configs()
            rows = [f"  {i + 1}. {c}" for i, c in enumerate(configs)]
            return "\n".join([f"plan: {len(configs)} config(s)"] + rows)

        keys = list(self._pipeline_axes.keys())
        all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*self._pipeline_axes.values())]
        pred = self._pipeline_config_filter_fn
        active = {frozenset(c.items()) for c in all_combos if pred is None or pred(c)}
        col_w = max(len(f"{k}={v}") for c in all_combos for k, v in c.items())
        rows = []
        for combo in all_combos:
            kept = frozenset(combo.items()) in active
            cells = "  ".join(f"{k}={str(v):<{col_w - len(k) - 1}}" for k, v in combo.items())
            rows.append(f"  {'○' if kept else '×'}  {cells}")
        total = len(all_combos)
        n_active = len(active)
        parts = [f"plan: {total} combinations ({n_active} active, {total - n_active} filtered)"] + rows

        if 2 <= len(keys) <= 3:
            parts += ["", self.grid()]

        return "\n".join(parts)

    def grid(self) -> str:
        if not self._pipeline_axes:
            raise ValueError("grid() requires at least one pipeline_config_axis()")
        axes = self._pipeline_axes
        keys = list(axes.keys())
        n = len(keys)
        if n < 2 or n > 3:
            raise ValueError(f"grid() requires 2 or 3 axes, got {n}")
        pred = self._pipeline_config_filter_fn
        all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]
        active = {frozenset(c.items()) for c in all_combos if pred is None or pred(c)}
        row_key, col_key = keys[0], keys[1]
        grp_key = keys[2] if n == 3 else None
        row_vals, col_vals = axes[row_key], axes[col_key]
        grp_vals = axes[grp_key] if grp_key else [None]
        cell_w = max(len(str(v)) for v in col_vals)
        row_label_w = max(len(str(v)) for v in row_vals)
        grp_w = len(col_vals) * (cell_w + 2) - 2
        axis_desc = f"row={row_key}  col={col_key}" + (f"  grp={grp_key}" if grp_key else "")
        lines = [f"grid: {axis_desc}", ""]
        if grp_key:
            grp_header = " " * (row_label_w + 2)
            for gi, gv in enumerate(grp_vals):
                grp_header += f"{str(gv):^{grp_w}}"
                if gi < len(grp_vals) - 1:
                    grp_header += "    "
            lines.append(grp_header)
        col_header = " " * (row_label_w + 2)
        block = "  ".join(f"{str(cv):^{cell_w}}" for cv in col_vals)
        col_header += "    ".join(block for _ in grp_vals)
        lines.append(col_header)
        for rv in row_vals:
            row = f"{str(rv):{row_label_w}}  "
            blocks = []
            for gv in grp_vals:
                cells = []
                for cv in col_vals:
                    combo = {row_key: rv, col_key: cv}
                    if grp_key:
                        combo[grp_key] = gv
                    mark = "○" if frozenset(combo.items()) in active else "×"
                    cells.append(f"{mark:^{cell_w}}")
                blocks.append("  ".join(cells))
            row += "    ".join(blocks)
            lines.append(row)
        lines += ["", "○ = active  × = filtered"]
        return "\n".join(lines)

    def _resolve_factory(self) -> PipelineFactory:
        if isinstance(self._source, Pipeline):
            _pipeline = self._source
            return lambda _: _pipeline
        return self._source

    def _resolve_pipeline_configs(self) -> list[dict]:
        if self._pipeline_axes:
            keys = list(self._pipeline_axes.keys())
            configs = [
                dict(zip(keys, combo))
                for combo in itertools.product(*self._pipeline_axes.values())
            ]
            if self._pipeline_config_filter_fn:
                configs = [c for c in configs if self._pipeline_config_filter_fn(c)]
            return configs
        if self._pipeline_configs_list is not None:
            return self._pipeline_configs_list
        return [self._pipeline_config_dict]

    def _resolve_data_factory(self) -> DataFactory:
        if self._data_input_fn is not None:
            _fn = self._data_input_fn
            return lambda _, fn=_fn: fn
        if self._data_factory_fn is not None:
            return self._data_factory_fn
        raise ValueError("no data_input() or data_factory() provided")

    def _resolve_data_configs(self) -> list[dict]:
        if self._data_input_fn is not None:
            return [{"_label": self._label_str or "input"}]
        if self._data_axes:
            keys = list(self._data_axes.keys())
            configs = [
                dict(zip(keys, combo))
                for combo in itertools.product(*self._data_axes.values())
            ]
            if self._data_config_filter_fn:
                configs = [c for c in configs if self._data_config_filter_fn(c)]
            return configs
        if self._data_configs_list is not None:
            return self._data_configs_list
        return [self._data_config_dict if self._data_config_dict else {}]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> list[BenchmarkResult]:
        """Execute the benchmark(s) and return all results."""
        self._validate()
        if verbose:
            print(self._plan(), file=sys.stderr)
            print(file=sys.stderr)
        return BenchmarkSweep(
            factory=self._resolve_factory(),
            configs=self._resolve_pipeline_configs(),
            data_factory=self._resolve_data_factory(),
            data_configs=self._resolve_data_configs(),
            measurement=self._build_measurement(),
        ).run()
