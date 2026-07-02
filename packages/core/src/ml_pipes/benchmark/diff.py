from __future__ import annotations

from dataclasses import dataclass

from .results import BenchmarkResult, InvocationStat


@dataclass(frozen=True)
class InvocationStatDiff:
    label: str
    only_in: str | None
    mean_delta_ms: float | None
    mean_delta_pct: float | None
    percentile_deltas: dict[float, float] | None


@dataclass(frozen=True)
class BenchmarkDiff:
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    total: InvocationStatDiff
    operators: list[InvocationStatDiff]

    def to_table(self) -> str:
        rows = [self.total, *self.operators]
        all_pct_keys: list[float] = []
        for row in rows:
            if row.percentile_deltas:
                for key in row.percentile_deltas:
                    if key not in all_pct_keys:
                        all_pct_keys.append(key)
        all_pct_keys.sort()
        pct_headers = [f"Δp{int(p * 100)}" for p in all_pct_keys]

        col_label = max(len(row.label) for row in rows)
        col_w = 12

        header = (
            f"{'operator':<{col_label}}  {'Δmean':>{col_w}}  {'Δmean%':>{col_w}}"
            + "".join(f"  {header:>{col_w}}" for header in pct_headers)
            + f"  {'note':>10}"
        )
        sep = "-" * len(header)

        def _fmt(value: float | None, suffix: str = "") -> str:
            if value is None:
                return "-"
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:.2f}{suffix}"

        lines = [
            f"baseline : {self.baseline.label}",
            f"candidate: {self.candidate.label}",
            sep,
            header,
            sep,
        ]
        for row in rows:
            note = f"only in {row.only_in}" if row.only_in else ""
            pct_cols = "".join(
                f"  {_fmt(row.percentile_deltas.get(pct) if row.percentile_deltas else None, 'ms'):>{col_w}}"
                for pct in all_pct_keys
            )
            line = (
                f"{row.label:<{col_label}}"
                f"  {_fmt(row.mean_delta_ms, 'ms'):>{col_w}}"
                f"  {_fmt(row.mean_delta_pct, '%'):>{col_w}}"
                + pct_cols
                + f"  {note:>10}"
            )
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)


def _span_diff(
    label: str,
    baseline: InvocationStat | None,
    candidate: InvocationStat | None,
) -> InvocationStatDiff:
    if baseline is None:
        return InvocationStatDiff(
            label=label,
            only_in="candidate",
            mean_delta_ms=None,
            mean_delta_pct=None,
            percentile_deltas=None,
        )
    if candidate is None:
        return InvocationStatDiff(
            label=label,
            only_in="baseline",
            mean_delta_ms=None,
            mean_delta_pct=None,
            percentile_deltas=None,
        )

    mean_delta = candidate.mean_ms - baseline.mean_ms
    mean_pct = (mean_delta / baseline.mean_ms * 100) if baseline.mean_ms != 0 else None

    shared_pct = set(baseline.percentiles) & set(candidate.percentiles)
    pct_deltas = {
        pct: candidate.percentiles[pct] - baseline.percentiles[pct]
        for pct in sorted(shared_pct)
    }

    return InvocationStatDiff(
        label=label,
        only_in=None,
        mean_delta_ms=mean_delta,
        mean_delta_pct=mean_pct,
        percentile_deltas=pct_deltas if pct_deltas else None,
    )


def _make_diff(baseline: BenchmarkResult, candidate: BenchmarkResult) -> BenchmarkDiff:
    total_diff = _span_diff("total", baseline.total, candidate.total)

    baseline_by_label = {stat.label: stat for stat in baseline.operators}
    candidate_by_label = {stat.label: stat for stat in candidate.operators}
    all_labels: list[str] = []
    seen: set[str] = set()
    for stat in baseline.operators:
        all_labels.append(stat.label)
        seen.add(stat.label)
    for stat in candidate.operators:
        if stat.label not in seen:
            all_labels.append(stat.label)

    span_diffs = [
        _span_diff(label, baseline_by_label.get(label), candidate_by_label.get(label))
        for label in all_labels
    ]
    return BenchmarkDiff(
        baseline=baseline,
        candidate=candidate,
        total=total_diff,
        operators=span_diffs,
    )
