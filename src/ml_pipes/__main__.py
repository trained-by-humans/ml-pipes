from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from ml_pipes.benchmark import (
    _DATA_FACTORY_ATTR,
    _PIPELINE_FACTORY_ATTR,
    Benchmark,
    BenchmarkMatrix,
    BenchmarkResult,
    BenchmarkSweep,
    InputFn,
    MeasurementConfig,
)


class CLIError(Exception):
    """User-facing CLI error — printed to stderr and exits with code 1."""


# ---------------------------------------------------------------------------
# Module / function loading
# ---------------------------------------------------------------------------

def _load_ref(ref: str) -> tuple[Any, Any]:
    """Parse 'pkg.module' or 'pkg.module:fn_name' and import the module.

    Returns (module, fn_or_None).
    """
    if ":" in ref:
        mod_path, fn_name = ref.rsplit(":", 1)
    else:
        mod_path, fn_name = ref, None

    try:
        module = importlib.import_module(mod_path)
    except ModuleNotFoundError as exc:
        raise CLIError(f"cannot import module {mod_path!r}: {exc}") from exc
    except Exception as exc:
        raise CLIError(f"error importing {mod_path!r}: {exc}") from exc

    if fn_name is None:
        return module, None

    fn = getattr(module, fn_name, None)
    if fn is None:
        public = [n for n in vars(module) if not n.startswith("_")]
        raise CLIError(
            f"function {fn_name!r} not found in {mod_path!r}. "
            f"Available names: {', '.join(public)}"
        )
    if not callable(fn):
        raise CLIError(f"{fn_name!r} in {mod_path!r} is not callable")
    return module, fn


def _discover_factory(
    module: Any, explicit_fn: Any, attr: str, kind: str
) -> Any:
    """Return explicit_fn if provided, otherwise scan module for attr marker.

    Returns None when nothing is found (caller decides whether that's an error).
    """
    if explicit_fn is not None:
        return explicit_fn

    found = [
        (name, fn)
        for name, fn in vars(module).items()
        if callable(fn) and getattr(fn, attr, False)
    ]

    if len(found) > 1:
        names = ", ".join(name for name, _ in found)
        raise CLIError(
            f"multiple @{kind}_factory found in {module.__name__!r}: [{names}]. "
            f"Only one is allowed per module; remove one or use 'module:{kind}_factory_fn' syntax."
        )

    return found[0][1] if found else None


def _resolve_pipeline_factory(module: Any, explicit_fn: Any, ref: str):
    fn = _discover_factory(module, explicit_fn, _PIPELINE_FACTORY_ATTR, "pipeline")
    if fn is None:
        mod_name = ref.split(":")[0]
        raise CLIError(
            f"no @pipeline_factory found in {mod_name!r}. "
            f"Decorate exactly one function with @pipeline_factory, "
            f"or use 'module:fn_name' syntax."
        )
    return fn


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _resolve_inputs(
    data_ref: str | None,
    input_paths: list[str] | None,
    pipeline_module: Any,
) -> tuple[list[InputFn], list[str]]:
    data_fn = None

    if data_ref is not None:
        data_module, explicit_fn = _load_ref(data_ref)
        data_fn = _discover_factory(data_module, explicit_fn, _DATA_FACTORY_ATTR, "data")
        if data_fn is None:
            raise CLIError(
                f"no @data_factory found in {data_ref!r}. "
                f"Decorate a function with @data_factory or use 'module:fn_name' syntax."
            )
    else:
        data_fn = _discover_factory(pipeline_module, None, _DATA_FACTORY_ATTR, "data")

    if data_fn is not None:
        input_fn = data_fn({})
        return [input_fn], [getattr(data_fn, "__wrapped__", data_fn).__name__]

    if not input_paths:
        raise CLIError(
            "no @data_factory found and no --input paths given. "
            "Either decorate a function with @data_factory, or pass --input path [...]."
        )
    return _build_file_input_fns(input_paths)


def _build_file_input_fns(paths: list[str]) -> tuple[list[InputFn], list[str]]:
    fns: list[InputFn] = []
    labels: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise CLIError(f"input file not found: {path}")
        def _make(p: Path = path) -> InputFn:
            def fn():
                return (p.name, p, None, None)
            return fn
        fns.append(_make())
        labels.append(path.name)
    return fns, labels


# ---------------------------------------------------------------------------
# Argument / config parsing
# ---------------------------------------------------------------------------

def _parse_axis_value(s: str) -> int | float | tuple | str:
    parts = s.split("x")
    if len(parts) > 1:
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            pass
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_arg_spec(spec: str) -> tuple[str, Any]:
    """Parse 'key=value' into (key, typed_value). Reuses axis value typing."""
    if "=" not in spec:
        raise CLIError(f"--arg must be in the form key=value — got: {spec!r}")
    key, _, value_str = spec.partition("=")
    key = key.strip()
    if not key:
        raise CLIError(f"--arg key is empty in: {spec!r}")
    return key, _parse_axis_value(value_str.strip())


def _parse_axis_spec(spec: str) -> tuple[str, list]:
    if "=" not in spec:
        raise CLIError(f"--axis must be in the form key=v1,v2,... — got: {spec!r}")
    key, _, values_str = spec.partition("=")
    key = key.strip()
    if not key:
        raise CLIError(f"--axis key is empty in: {spec!r}")
    values = [_parse_axis_value(v.strip()) for v in values_str.split(",") if v.strip()]
    if not values:
        raise CLIError(f"--axis has no values in: {spec!r}")
    return key, values


def _parse_configs(raw_configs: list[str]) -> list[dict]:
    result = []
    for i, raw in enumerate(raw_configs):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CLIError(
                f"invalid JSON in --config #{i + 1}: {exc}\n  value was: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise CLIError(
                f"--config #{i + 1} must be a JSON object (dict), "
                f"got {type(parsed).__name__}: {raw!r}"
            )
        result.append(parsed)
    return result


def _build_measurement(args: argparse.Namespace) -> MeasurementConfig:
    warmup = args.warmup if args.warmup is not None else max(5, args.runs // 10)
    return MeasurementConfig(
        runs=args.runs,
        warmup=warmup,
        percentiles=tuple(args.percentiles),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(factory, config: dict) -> None:
    """Raise CLIError if config is missing required parameters."""
    wrapped = getattr(factory, "__wrapped__", None)
    if wrapped is None:
        return
    try:
        inspect.signature(wrapped).bind(**config)
    except TypeError as exc:
        raise CLIError(f"missing required argument for config {config}: {exc}") from exc


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_results(results: list[BenchmarkResult], save_dir: str) -> None:
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = out / result.slug(".json")
        result.save(str(path))
        print(f"saved: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> int:
    module, explicit_fn = _load_ref(args.pipeline_ref)
    factory = _resolve_pipeline_factory(module, explicit_fn, args.pipeline_ref)
    input_fns, input_labels = _resolve_inputs(
        data_ref=getattr(args, "data_ref", None),
        input_paths=args.input,
        pipeline_module=module,
    )
    measurement = _build_measurement(args)
    expand_regions = not args.collapse_regions

    config = dict(_parse_arg_spec(s) for s in (args.args or []))

    results = []
    for input_fn, label in zip(input_fns, input_labels):
        _validate_config(factory, config)
        pipeline = factory(config)
        result = Benchmark(
            pipeline=pipeline,
            input_fn=input_fn,
            measurement=measurement,
            label=label,
        ).run()
        results.append(result)

    if len(results) == 1:
        print(results[0].to_table(expand_regions=expand_regions))
    else:
        print(BenchmarkResult.to_comparison_table(results, expand_regions=expand_regions))

    if args.save:
        _save_results(results, args.save)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    module, explicit_fn = _load_ref(args.pipeline_ref)
    factory = _resolve_pipeline_factory(module, explicit_fn, args.pipeline_ref)
    input_fns, input_labels = _resolve_inputs(
        data_ref=getattr(args, "data_ref", None),
        input_paths=args.input,
        pipeline_module=module,
    )
    measurement = _build_measurement(args)
    expand_regions = not args.collapse_regions

    if args.axes:
        axes: dict[str, list] = {}
        for spec in args.axes:
            key, values = _parse_axis_spec(spec)
            axes[key] = values
        runner = BenchmarkMatrix(
            factory=factory,
            axes=axes,
            input_fns=input_fns,
            input_labels=input_labels,
            measurement=measurement,
        )
        print(runner.to_plan(), file=sys.stderr)
        print(file=sys.stderr)
        results = runner.run()

    elif args.configs:
        configs = _parse_configs(args.configs)
        for c in configs:
            _validate_config(factory, c)
        results = BenchmarkSweep(
            factory=factory,
            configs=configs,
            input_fns=input_fns,
            input_labels=input_labels,
            measurement=measurement,
        ).run()

    else:
        _validate_config(factory, {})
        results = BenchmarkSweep(
            factory=factory,
            configs=[{}],
            input_fns=input_fns,
            input_labels=input_labels,
            measurement=measurement,
        ).run()

    print(BenchmarkResult.to_comparison_table(results, expand_regions=expand_regions))

    if args.save:
        _save_results(results, args.save)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-pipes",
        description="Run benchmarks against an ml-pipes pipeline module.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--runs", type=int, default=100, metavar="N",
                        help="Measured runs (default: 100)")
    shared.add_argument("--warmup", type=int, default=None, metavar="N",
                        help="Warmup runs (default: max(5, runs//10))")
    shared.add_argument("--percentiles", type=float, nargs="+",
                        default=[0.50, 0.95, 0.99], metavar="P",
                        help="Percentiles to compute (default: 0.5 0.95 0.99)")
    shared.add_argument("--save", metavar="DIR", default=None,
                        help="Directory to save per-result JSON files")
    shared.add_argument("--collapse-regions", action="store_true", default=False,
                        help="Collapse region spans in output table")

    bench_p = sub.add_parser(
        "benchmark", parents=[shared],
        help="Single benchmark run with default config",
        description="Discover @pipeline_factory and @data_factory and run a single benchmark.",
    )
    bench_p.add_argument("pipeline_ref", metavar="MODULE[:FN]",
                         help="Pipeline factory: 'pkg.module' or 'pkg.module:fn_name'")
    bench_p.add_argument("data_ref", metavar="DATA_MODULE[:FN]", nargs="?", default=None,
                         help="Data factory (auto-discovered in pipeline module if absent)")
    bench_p.add_argument("--input", nargs="+", metavar="PATH",
                         help="Fallback input files when no @data_factory is present")
    bench_p.add_argument("--arg", action="append", dest="args", metavar="KEY=VALUE",
                         help="Factory argument (repeatable), e.g. --arg max_concurrency=8")

    sweep_p = sub.add_parser(
        "sweep", parents=[shared],
        help="Parameterized sweep over configs or axes",
        description=(
            "Run multiple benchmark configurations. "
            "Use --config for explicit JSON dicts or --axis for cartesian-product expansion."
        ),
    )
    sweep_p.add_argument("pipeline_ref", metavar="MODULE[:FN]",
                         help="Pipeline factory reference")
    sweep_p.add_argument("data_ref", metavar="DATA_MODULE[:FN]", nargs="?", default=None,
                         help="Data factory reference (auto-discovered if absent)")
    sweep_p.add_argument("--input", nargs="+", metavar="PATH",
                         help="Fallback input files")
    config_group = sweep_p.add_mutually_exclusive_group()
    config_group.add_argument("--config", action="append", dest="configs", metavar="JSON",
                              help="Explicit config as JSON dict (repeatable)")
    config_group.add_argument("--axis", action="append", dest="axes",
                              metavar="KEY=V1,V2,...",
                              help="Axis for cartesian expansion (repeatable)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "benchmark":
            exit_code = cmd_benchmark(args)
        elif args.command == "sweep":
            exit_code = cmd_sweep(args)
        else:
            parser.print_help()
            exit_code = 1
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
