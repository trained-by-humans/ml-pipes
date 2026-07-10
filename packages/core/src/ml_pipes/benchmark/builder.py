from __future__ import annotations

import itertools
import sys
from typing import Any, Callable, TypeAlias

from ml_pipes.pipeline import Pipeline
from ml_pipes.factory import DataFactory, InputFn, PipelineFactory
from ml_pipes.benchmark.results import BenchmarkResult
from ml_pipes.benchmark.runner import BenchmarkSweep, MeasurementConfig

PipelineFactoryLike: TypeAlias = Callable[..., Pipeline[Any, Any]]
DataFactoryLike: TypeAlias = Callable[..., InputFn]
ConfigFilter: TypeAlias = Callable[[dict], bool]


class BenchmarkBuilder:
    """Fluent builder for benchmarks.

    Entry points::

        BenchmarkBuilder.pipeline(p)   # concrete Pipeline (no config sweep)
        BenchmarkBuilder.factory(f)    # factory callable (config sweep)

    Factories can be decorated reusable functions or plain callables invoked
    as ``fn(**config)``.

    Chain measurement, pipeline config, and data config methods, then call
    ``.run()`` which returns ``list[BenchmarkResult]``.
    """

    def __init__(self, source: Pipeline[Any, Any] | PipelineFactoryLike) -> None:
        self._pipeline_source = (
            source if isinstance(source, Pipeline) else PipelineFactory.ensure_factory(source)
        )

        self._input_fn: InputFn | None = None
        self._data_factory: DataFactory | None = None

        self._pipeline_config_dict: dict = {}
        self._pipeline_config_set: list[dict] | None = None
        self._pipeline_axes: dict[str, list] = {}
        self._pipeline_config_filter: Callable | None = None

        self._data_config_dict: dict = {}
        self._data_config_set: list[dict] | None = None
        self._data_axes: dict[str, list] = {}
        self._data_config_filter: Callable | None = None

        self._runs: int | None = None
        self._warmup: int | None = None
        self._percentiles: list[float] | None = None
        self._label: str | None = None
        self._metadata: dict | None = None

    @classmethod
    def pipeline(cls, p: Pipeline[Any, Any]) -> "BenchmarkBuilder":
        """Start from a concrete Pipeline (no config sweep)."""
        return cls(p)

    @classmethod
    def factory(cls, f: PipelineFactoryLike) -> "BenchmarkBuilder":
        """Start from a pipeline factory or callable."""
        return cls(f)

    def pipeline_config(self, **kwargs) -> "BenchmarkBuilder":
        """Set a single pipeline config dict."""
        self._pipeline_config_dict.update(kwargs)
        return self

    def pipeline_config_set(self, configs: list[dict]) -> "BenchmarkBuilder":
        """Set an explicit list of handpicked pipeline configs for a sweep."""
        self._pipeline_config_set = list(configs)
        return self

    def pipeline_config_axis(self, key: str, *values) -> "BenchmarkBuilder":
        """Register a pipeline config axis for cartesian expansion."""
        self._pipeline_axes[key] = list(values)
        return self

    def pipeline_config_filter(self, pred: ConfigFilter) -> "BenchmarkBuilder":
        """Drop pipeline configs where pred returns False."""
        self._pipeline_config_filter = pred
        return self

    def data_input(self, fn: InputFn) -> "BenchmarkBuilder":
        """Use a concrete InputFn (no data config sweep)."""
        self._input_fn = fn
        return self

    def data_inputs(self, fns: list[InputFn], labels: list[str]) -> "BenchmarkBuilder":
        """Use multiple InputFns as a sweep — each gets a label."""
        if len(fns) != len(labels):
            raise ValueError("data_inputs() requires the same number of input functions and labels")
        if len(set(labels)) != len(labels):
            raise ValueError("data_inputs() requires unique labels")

        fn_map = dict(zip(labels, fns))

        def _select_input(_label: str) -> InputFn:
            return fn_map[_label]

        self._data_factory = DataFactory.from_callable(_select_input)
        self._data_config_set = [{"_label": label} for label in labels]
        return self

    def data_factory(self, factory: DataFactoryLike) -> "BenchmarkBuilder":
        """Use a data factory or callable (enables data config sweep)."""
        self._data_factory = DataFactory.ensure_factory(factory)
        return self

    def data_config(self, **kwargs) -> "BenchmarkBuilder":
        """Set a single data config dict."""
        self._data_config_dict.update(kwargs)
        return self

    def data_config_set(self, configs: list[dict]) -> "BenchmarkBuilder":
        """Set an explicit list of handpicked data configs for a sweep."""
        self._data_config_set = list(configs)
        return self

    def data_config_axis(self, key: str, *values) -> "BenchmarkBuilder":
        """Register a data config axis for cartesian expansion."""
        self._data_axes[key] = list(values)
        return self

    def data_config_filter(self, pred: ConfigFilter) -> "BenchmarkBuilder":
        """Drop data configs where pred returns False."""
        self._data_config_filter = pred
        return self

    def runs(self, n: int) -> "BenchmarkBuilder":
        self._runs = n
        return self

    def warmup(self, n: int) -> "BenchmarkBuilder":
        self._warmup = n
        return self

    def percentiles(self, *ps: float) -> "BenchmarkBuilder":
        self._percentiles = list(ps)
        return self

    def label(self, s: str) -> "BenchmarkBuilder":
        self._label = s
        return self

    def metadata(self, d: dict) -> "BenchmarkBuilder":
        self._metadata = d
        return self

    def _build_measurement(self) -> MeasurementConfig:
        runs = self._runs or 100
        warmup = self._warmup if self._warmup is not None else max(5, runs // 10)
        percentiles = tuple(self._percentiles) if self._percentiles else (0.50, 0.95, 0.99)
        return MeasurementConfig(runs=runs, warmup=warmup, percentiles=percentiles)

    def _validate(self) -> None:
        self._validate_pipeline_source()
        self._validate_data_source()

    def _validate_pipeline_source(self) -> None:
        if isinstance(self._pipeline_source, Pipeline):
            has_pipeline_config = (
                self._pipeline_config_dict
                or self._pipeline_config_set is not None
                or self._pipeline_axes
            )
            if has_pipeline_config:
                raise ValueError(
                    "pipeline config methods cannot be used with BenchmarkBuilder.pipeline() — "
                    "a concrete Pipeline ignores config. Use BenchmarkBuilder.factory() instead."
                )
        if self._pipeline_config_set is not None and self._pipeline_axes:
            raise ValueError("pipeline_config_set() and pipeline_config_axis() are mutually exclusive")

    def _validate_data_source(self) -> None:
        if self._input_fn is not None and self._data_factory is not None:
            raise ValueError("data_input() and data_factory() are mutually exclusive")
        has_data_config = (
            self._data_config_dict
            or self._data_config_set is not None
            or self._data_axes
        )
        if has_data_config and self._input_fn is not None:
            raise ValueError(
                "data config methods cannot be used with data_input() — "
                "a concrete InputFn ignores config. Use data_factory() instead."
            )
        if has_data_config and self._data_factory is None and self._input_fn is None:
            raise ValueError("data_config*() requires data_factory() to be set")
        if self._data_config_set is not None and self._data_axes:
            raise ValueError("data_config_set() and data_config_axis() are mutually exclusive")

    def plan(self) -> str:
        output = self._render_plan()
        print(output, file=sys.stderr)
        return output

    def _render_plan(self) -> str:
        if not self._pipeline_axes:
            configs = self._resolve_pipeline_configs()
            rows = [f"  {i + 1}. {config}" for i, config in enumerate(configs)]
            return "\n".join([f"plan: {len(configs)} config(s)"] + rows)

        keys = list(self._pipeline_axes.keys())
        all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*self._pipeline_axes.values())]
        pred = self._pipeline_config_filter
        active = {frozenset(config.items()) for config in all_combos if pred is None or pred(config)}
        col_w = max(len(f"{key}={value}") for config in all_combos for key, value in config.items())
        rows = []
        for combo in all_combos:
            kept = frozenset(combo.items()) in active
            cells = "  ".join(f"{key}={str(value):<{col_w - len(key) - 1}}" for key, value in combo.items())
            rows.append(f"  {'○' if kept else '×'}  {cells}")
        total = len(all_combos)
        active_count = len(active)
        parts = [f"plan: {total} combinations ({active_count} active, {total - active_count} filtered)"] + rows

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
        pred = self._pipeline_config_filter
        all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]
        active = {frozenset(config.items()) for config in all_combos if pred is None or pred(config)}
        row_key, col_key = keys[0], keys[1]
        grp_key = keys[2] if n == 3 else None
        row_vals, col_vals = axes[row_key], axes[col_key]
        grp_vals = axes[grp_key] if grp_key else [None]
        cell_w = max(len(str(value)) for value in col_vals)
        row_label_w = max(len(str(value)) for value in row_vals)
        grp_w = len(col_vals) * (cell_w + 2) - 2
        axis_desc = f"row={row_key}  col={col_key}" + (f"  grp={grp_key}" if grp_key else "")
        lines = [f"grid: {axis_desc}", ""]
        if grp_key:
            grp_header = " " * (row_label_w + 2)
            for group_index, group_value in enumerate(grp_vals):
                grp_header += f"{str(group_value):^{grp_w}}"
                if group_index < len(grp_vals) - 1:
                    grp_header += "    "
            lines.append(grp_header)
        col_header = " " * (row_label_w + 2)
        block = "  ".join(f"{str(col_value):^{cell_w}}" for col_value in col_vals)
        col_header += "    ".join(block for _ in grp_vals)
        lines.append(col_header)
        for row_value in row_vals:
            row = f"{str(row_value):{row_label_w}}  "
            blocks = []
            for group_value in grp_vals:
                cells = []
                for col_value in col_vals:
                    combo = {row_key: row_value, col_key: col_value}
                    if grp_key:
                        combo[grp_key] = group_value
                    mark = "○" if frozenset(combo.items()) in active else "×"
                    cells.append(f"{mark:^{cell_w}}")
                blocks.append("  ".join(cells))
            row += "    ".join(blocks)
            lines.append(row)
        lines += ["", "○ = active  × = filtered"]
        return "\n".join(lines)

    def _resolve_pipeline_factory(self) -> PipelineFactory[Any, Any]:
        if isinstance(self._pipeline_source, Pipeline):
            pipeline = self._pipeline_source
            return PipelineFactory.from_callable(lambda: pipeline)
        return self._pipeline_source

    def _resolve_pipeline_configs(self) -> list[dict]:
        if self._pipeline_axes:
            keys = list(self._pipeline_axes.keys())
            configs = [
                {**self._pipeline_config_dict, **dict(zip(keys, combo))}
                for combo in itertools.product(*self._pipeline_axes.values())
            ]
            if self._pipeline_config_filter:
                configs = [config for config in configs if self._pipeline_config_filter(config)]
            return configs
        if self._pipeline_config_set is not None:
            return [{**self._pipeline_config_dict, **config} for config in self._pipeline_config_set]
        return [self._pipeline_config_dict]

    def _resolve_data_factory(self) -> DataFactory:
        if self._input_fn is not None:
            input_fn = self._input_fn
            return DataFactory.from_callable(lambda: input_fn)
        if self._data_factory is not None:
            return self._data_factory
        raise ValueError("no data_input() or data_factory() provided")

    def _resolve_data_configs(self) -> list[dict]:
        if self._input_fn is not None:
            return [{}]
        if self._data_axes:
            keys = list(self._data_axes.keys())
            configs = [
                {**self._data_config_dict, **dict(zip(keys, combo))}
                for combo in itertools.product(*self._data_axes.values())
            ]
            if self._data_config_filter:
                configs = [config for config in configs if self._data_config_filter(config)]
            return configs
        if self._data_config_set is not None:
            return [{**self._data_config_dict, **config} for config in self._data_config_set]
        return [self._data_config_dict if self._data_config_dict else {}]

    def run(self, verbose: bool = True) -> list[BenchmarkResult]:
        """Execute the benchmark(s) and return all results."""
        self._validate()
        if verbose:
            print(self._render_plan(), file=sys.stderr)
            print(file=sys.stderr)
        return BenchmarkSweep(
            factory=self._resolve_pipeline_factory(),
            configs=self._resolve_pipeline_configs(),
            data_factory=self._resolve_data_factory(),
            data_configs=self._resolve_data_configs(),
            measurement=self._build_measurement(),
            label_prefix=self._label or None,
            extra_metadata=self._metadata or None,
        ).run()
