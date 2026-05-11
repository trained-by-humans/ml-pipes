from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .collectors.concurrent_collector import ConcurrentCollector
from .core import Pipeline
from .tracing import InvocationTrace


@dataclass(frozen=True)
class InvocationStat:
    """Latency statistics for one operator or the whole pipeline across N runs."""

    label: str
    count: int
    mean_ms: float
    stddev_ms: float        # run-to-run jitter; high value → noisy operator
    min_ms: float
    max_ms: float
    percentiles: dict[float, float]  # keys mirror MeasurementConfig.percentiles exactly


def _compute_span(label: str, samples_s: list[float], percentiles: tuple[float, ...]) -> InvocationStat:
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
    operators: list[InvocationStat]   # per-operator latency breakdown — only ml-pipes can produce this

    def to_table(self) -> str:
        rows = [self.total, *self.operators]
        pct_keys = sorted(self.total.percentiles)
        pct_headers = [f"p{int(p * 100)}" for p in pct_keys]

        col_label = max(len(r.label) for r in rows)
        col_w = 9

        header = (
            f"{'operator':<{col_label}}  {'mean':>{col_w}}"
            + "".join(f"  {h:>{col_w}}" for h in pct_headers)
            + f"  {'stddev':>{col_w}}  {'min':>{col_w}}  {'max':>{col_w}}"
        )
        sep = "-" * len(header)

        lines = [header, sep]
        for r in rows:
            line = (
                f"{r.label:<{col_label}}  {r.mean_ms:>{col_w}.2f}"
                + "".join(f"  {r.percentiles[p]:>{col_w}.2f}" for p in pct_keys)
                + f"  {r.stddev_ms:>{col_w}.2f}  {r.min_ms:>{col_w}.2f}  {r.max_ms:>{col_w}.2f}"
            )
            lines.append(line)
        lines.append(sep)
        lines.append(f"runs: {self.total.count}  (all values in ms)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def _span(s: InvocationStat) -> dict:
            return {
                "label": s.label,
                "count": s.count,
                "mean_ms": s.mean_ms,
                "stddev_ms": s.stddev_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
                "percentiles": {str(k): v for k, v in s.percentiles.items()},
            }

        return {
            "label": self.label,
            "metadata": self.metadata,
            "total": _span(self.total),
            "operators": [_span(s) for s in self.operators],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> BenchmarkResult:
        with open(path) as f:
            d = json.load(f)

        def _span(s: dict) -> InvocationStat:
            return InvocationStat(
                label=s["label"],
                count=s["count"],
                mean_ms=s["mean_ms"],
                stddev_ms=s["stddev_ms"],
                min_ms=s["min_ms"],
                max_ms=s["max_ms"],
                percentiles={float(k): v for k, v in s["percentiles"].items()},
            )

        return cls(
            label=d["label"],
            metadata=d.get("metadata", {}),
            total=_span(d["total"]),
            operators=[_span(s) for s in d["operators"]],
        )

    def diff(self, other: BenchmarkResult) -> BenchmarkDiff:
        return _make_diff(self, other)


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
        self._op_samples: dict[str, list[float]] = {}

    def _collect(self, trace: InvocationTrace) -> None:
        self._calls += 1
        if self._calls <= self._config.warmup:
            return
        self._total_samples.append(trace.total_duration_s)
        for span in trace.spans:
            self._op_samples.setdefault(span.label, []).append(span.duration_s)

    def report(self, label: str = "", metadata: dict | None = None) -> BenchmarkResult:
        self.flush()
        pct = self._config.percentiles
        total = _compute_span("total", self._total_samples, pct) if self._total_samples else _compute_span("total", [0.0], pct)
        operators = [_compute_span(lbl, samples, pct) for lbl, samples in self._op_samples.items()]
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
        self._op_samples.clear()


# InputFn returns (id, value, tag, metadata).
# tag and metadata are reserved for future bucketing/annotation features and ignored for now.
InputFn = Callable[[], tuple[str, Any, str | None, dict | None]]


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
        config: MeasurementConfig = MeasurementConfig(),
        label: str = "",
        metadata: dict | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._input_fn = input_fn
        self._config = config
        self._label = label
        self._metadata = metadata or {}

    def run(self) -> BenchmarkResult:
        prior_tracing = self._pipeline._tracing_config
        collector = BenchmarkCollector(self._config)
        self._pipeline.set_tracing(collector)
        try:
            total_runs = self._config.warmup + self._config.runs
            for _ in range(total_runs):
                _id, value, _tag, _meta = self._input_fn()
                self._pipeline(value)
        finally:
            collector.stop()
            self._pipeline._tracing_config = prior_tracing

        return collector.report(label=self._label, metadata=self._metadata)
