from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_pipes.benchmark.diff import BenchmarkDiff


@dataclass
class InvocationStat:
    """Latency statistics for one operator or the whole pipeline across N runs.

    Children mirror StepSpan.child_trace: present when the operator is a region
    (e.g. Scatter) and child spans were collected.
    """

    label: str
    count: int
    mean_ms: float
    stddev_ms: float        # run-to-run jitter; high value -> noisy operator
    min_ms: float
    max_ms: float
    percentiles: dict[float, float]
    children: list["InvocationStat"] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Portable benchmark artifact. `operators` is the ml-pipes-specific per-operator breakdown."""

    label: str
    metadata: dict
    total: InvocationStat
    operators: list[InvocationStat]

    def to_table(self, expand_regions: bool = True) -> str:
        pct_keys = sorted(self.total.percentiles)
        pct_headers = [f"p{int(p * 100)}" for p in pct_keys]

        flat = [(0, self.total)] + _flat_stats(self.operators, expand_regions)

        col_label = max(len("  " * depth + stat.label) for depth, stat in flat)
        col_w = 9

        header = (
            f"{'operator':<{col_label}}   {'mean':>{col_w}}"
            + "".join(f"  {header:>{col_w}}" for header in pct_headers)
            + f"  {'stddev':>{col_w}}  {'min':>{col_w}}  {'max':>{col_w}}"
        )
        sep = "-" * len(header)

        lines = [header, sep]
        any_collapsed = False
        for depth, stat in flat:
            indent = "  " * depth
            collapsed = not expand_regions and bool(stat.children)
            if collapsed:
                any_collapsed = True
            label = indent + stat.label + ("*" if collapsed else "")
            line = (
                f"{label:<{col_label}}  {stat.mean_ms:>{col_w}.2f}"
                + "".join(f"  {stat.percentiles[p]:>{col_w}.2f}" for p in pct_keys)
                + f"  {stat.stddev_ms:>{col_w}.2f}  {stat.min_ms:>{col_w}.2f}  {stat.max_ms:>{col_w}.2f}"
            )
            lines.append(line)
        lines.append(sep)
        footer = f"runs: {self.total.count}  (all values in ms)"
        if any_collapsed:
            footer += "\n* Child spans are collapsed"
        lines.append(footer)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def _stat(stat: InvocationStat) -> dict:
            data: dict = {
                "label": stat.label,
                "count": stat.count,
                "mean_ms": stat.mean_ms,
                "stddev_ms": stat.stddev_ms,
                "min_ms": stat.min_ms,
                "max_ms": stat.max_ms,
                "percentiles": {str(key): value for key, value in stat.percentiles.items()},
            }
            if stat.children:
                data["children"] = [_stat(child) for child in stat.children]
            return data

        return {
            "label": self.label,
            "metadata": self.metadata,
            "total": _stat(self.total),
            "operators": [_stat(stat) for stat in self.operators],
        }

    def slug(self, ext: str = "") -> str:
        safe = re.sub(r'[/\\:*?"<>|]', "_", self.label)
        return safe + ext

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        with open(path) as file:
            data = json.load(file)

        def _stat(stat: dict) -> InvocationStat:
            return InvocationStat(
                label=stat["label"],
                count=stat["count"],
                mean_ms=stat["mean_ms"],
                stddev_ms=stat["stddev_ms"],
                min_ms=stat["min_ms"],
                max_ms=stat["max_ms"],
                percentiles={float(key): value for key, value in stat["percentiles"].items()},
                children=[_stat(child) for child in stat.get("children", [])],
            )

        return cls(
            label=data["label"],
            metadata=data.get("metadata", {}),
            total=_stat(data["total"]),
            operators=[_stat(stat) for stat in data["operators"]],
        )

    def diff(self, other: "BenchmarkResult") -> "BenchmarkDiff":
        from ml_pipes.benchmark.diff import _make_diff

        return _make_diff(self, other)

    @staticmethod
    def to_comparison_table(results: list["BenchmarkResult"], expand_regions: bool = True) -> str:
        """Render a multi-column comparison table: one column per result, rows are operators."""
        if not results:
            return "(no results)"

        all_rows: list[tuple[int, str]] = [(0, "total")]
        seen: set[str] = {"total"}
        for result in results:
            for depth, stat in _flat_stats(result.operators, expand_regions):
                if stat.label not in seen:
                    all_rows.append((depth, stat.label))
                    seen.add(stat.label)

        all_pct: set[float] = set()
        for result in results:
            all_pct.update(result.total.percentiles)
        pct_keys = sorted(all_pct)

        col_label = max(len("  " * depth + label) for depth, label in all_rows)
        col_w = 9
        cols_per_result = 1 + len(pct_keys)
        result_col_w = cols_per_result * (col_w + 2)

        def _header_for(result: BenchmarkResult) -> str:
            label = result.label
            width = result_col_w - 1
            if len(label) > width:
                if "|" in label:
                    input_part, config_part = label.split("|", 1)
                    if len(config_part) <= width - 2:
                        keep = width - len(config_part) - 1
                        label = input_part[:keep] + "…|" + config_part
                    else:
                        label = config_part[-width:]
                else:
                    half = (width - 1) // 2
                    label = label[:half] + "…" + label[len(label) - (width - half - 1):]
            return f"{label:<{result_col_w}}"

        def _subheader() -> str:
            sub = f"{'mean':>{col_w}}"
            for pct in pct_keys:
                sub += f"  {f'p{int(pct * 100)}':>{col_w}}"
            return sub

        def _row_for(stat: InvocationStat | None) -> str:
            if stat is None:
                return " " * col_w + "  " + "  ".join("-" * col_w for _ in pct_keys)
            row = f"{stat.mean_ms:>{col_w}.2f}"
            for pct in pct_keys:
                if pct in stat.percentiles:
                    row += f"  {stat.percentiles[pct]:>{col_w}.2f}"
                else:
                    row += f"  {'-':>{col_w}}"
            return row

        lookups: list[dict[str, InvocationStat]] = []
        for result in results:
            lookup: dict[str, InvocationStat] = {"total": result.total}
            lookup.update({stat.label: stat for _, stat in _flat_stats(result.operators, expand_regions=True)})
            lookups.append(lookup)

        sep_width = col_label + 3 + len(results) * (result_col_w + 2)
        sep = "-" * sep_width

        lines = [sep]
        lines.append(" " * (col_label + 3) + "  ".join(_header_for(result) for result in results))
        lines.append(" " * (col_label + 3) + "  ".join(_subheader() for _ in results))
        lines.append(sep)

        any_collapsed = False
        for depth, label in all_rows:
            indent = "  " * depth
            collapsed = not expand_regions and any(
                (stat := lookup.get(label)) is not None and bool(stat.children)
                for lookup in lookups
            )
            if collapsed:
                any_collapsed = True
            display = indent + label + ("*" if collapsed else "")
            row = f"{display:<{col_label}}  "
            row += "  ".join(_row_for(lookup.get(label)) for lookup in lookups)
            lines.append(row)

        lines.append(sep)
        footer = f"runs: {results[0].total.count}  (all values in ms)"
        if any_collapsed:
            footer += "\n* Child spans are collapsed"
        lines.append(footer)
        return "\n".join(lines)


def _flat_stats(
    stats: list[InvocationStat],
    expand_regions: bool = True,
    depth: int = 0,
) -> list[tuple[int, InvocationStat]]:
    rows: list[tuple[int, InvocationStat]] = []
    for stat in stats:
        rows.append((depth, stat))
        if expand_regions and stat.children:
            rows.extend(_flat_stats(stat.children, expand_regions, depth + 1))
    return rows
