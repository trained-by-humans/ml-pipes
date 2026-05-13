from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from ml_pipes.benchmark import (
    Benchmark,
    BenchmarkMatrix,
    BenchmarkResult,
    BenchmarkSweep,
    MeasurementConfig,
)
from ml_pipes.factory import (
    _DATA_FACTORY_ATTR,
    _PIPELINE_FACTORY_ATTR,
    InputFn,
    discover_factory,
    validate_factory_config,
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


def _resolve_pipeline_factory(module: Any, explicit_fn: Any, ref: str):
    try:
        fn = discover_factory(module, explicit_fn, _PIPELINE_FACTORY_ATTR, "pipeline")
    except ValueError as exc:
        raise CLIError(str(exc)) from exc
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

def _resolve_data_factory(
    data_ref: str | None,
    pipeline_module: Any,
) -> Any:
    """Discover and return the data factory callable without calling it."""
    if data_ref is not None:
        data_module, explicit_fn = _load_ref(data_ref)
        try:
            data_fn = discover_factory(data_module, explicit_fn, _DATA_FACTORY_ATTR, "data")
        except ValueError as exc:
            raise CLIError(str(exc)) from exc
        if data_fn is None:
            raise CLIError(
                f"no @data_factory found in {data_ref!r}. "
                f"Decorate a function with @data_factory or use 'module:fn_name' syntax."
            )
        return data_fn

    try:
        return discover_factory(pipeline_module, None, _DATA_FACTORY_ATTR, "data")
    except ValueError as exc:
        raise CLIError(str(exc)) from exc


def _resolve_inputs(
    data_ref: str | None,
    input_paths: list[str] | None,
    pipeline_module: Any,
    data_config: dict,
) -> tuple[list[InputFn], list[str]]:
    data_fn = _resolve_data_factory(data_ref, pipeline_module)

    if data_fn is not None:
        input_fn = data_fn(data_config)
        return [input_fn], [getattr(data_fn, "__wrapped__", data_fn).__name__]

    if not input_paths:
        raise CLIError(
            "no @data_factory found and no --input paths given. "
            "Either decorate a function with @data_factory, or pass --input path [...]."
        )
    return _build_file_input_fns(input_paths)


def _build_file_input_fns(paths: list[str]) -> tuple[list[InputFn], list[str]]:
    resolved = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise CLIError(f"input file not found: {path}")
        resolved.append(path)

    basenames = [p.name for p in resolved]
    duplicate = {name for name in basenames if basenames.count(name) > 1}

    fns: list[InputFn] = []
    labels: list[str] = []
    for path in resolved:
        label = str(path) if path.name in duplicate else path.name
        def _make(p: Path = path, id_: str = label) -> InputFn:
            def fn():
                return (id_, p, None, None)
            return fn
        fns.append(_make())
        labels.append(label)
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


def _parse_args_to_config(specs: list[str], flag: str) -> dict:
    """Parse repeated KEY=VALUE specs into a dict, raising on duplicate keys."""
    config: dict = {}
    for spec in specs:
        key, value = _parse_arg_spec(spec)
        if key in config:
            raise CLIError(f"{flag} key {key!r} specified more than once")
        config[key] = value
    return config


def _validate_config(factory, config: dict) -> None:
    try:
        validate_factory_config(factory, config)
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

def cmd_run(args: argparse.Namespace) -> int:
    module, explicit_fn = _load_ref(args.pipeline_ref)
    factory = _resolve_pipeline_factory(module, explicit_fn, args.pipeline_ref)
    pipeline_config = _parse_args_to_config(args.args or [], "--arg")
    data_config = _parse_args_to_config(args.data_args or [], "--data-arg")
    input_fns, _ = _resolve_inputs(
        data_ref=getattr(args, "data_ref", None),
        input_paths=args.input,
        pipeline_module=module,
        data_config=data_config,
    )
    _validate_config(factory, pipeline_config)
    pipeline = factory(pipeline_config)
    for input_fn in input_fns:
        _, value, _, _ = input_fn()
        pipeline(value)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    module, explicit_fn = _load_ref(args.pipeline_ref)
    factory = _resolve_pipeline_factory(module, explicit_fn, args.pipeline_ref)
    pipeline_config = _parse_args_to_config(args.args or [], "--arg")
    data_config = _parse_args_to_config(args.data_args or [], "--data-arg")
    input_fns, input_labels = _resolve_inputs(
        data_ref=getattr(args, "data_ref", None),
        input_paths=args.input,
        pipeline_module=module,
        data_config=data_config,
    )
    measurement = _build_measurement(args)
    expand_regions = not args.collapse_regions

    results = []
    for input_fn, label in zip(input_fns, input_labels):
        _validate_config(factory, pipeline_config)
        pipeline = factory(pipeline_config)
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
    data_fn = _resolve_data_factory(getattr(args, "data_ref", None), module)
    measurement = _build_measurement(args)
    expand_regions = not args.collapse_regions

    # Pipeline dimension
    if args.axes:
        pipeline_axes = {}
        for spec in args.axes:
            key, values = _parse_axis_spec(spec)
            pipeline_axes[key] = values
    elif args.configs:
        pipeline_configs = _parse_configs(args.configs)
    else:
        pipeline_configs = [_parse_args_to_config(args.args or [], "--arg")]

    # Data dimension
    if args.data_axes:
        data_axes = {}
        for spec in args.data_axes:
            key, values = _parse_axis_spec(spec)
            data_axes[key] = values
    elif args.data_configs:
        data_configs = _parse_configs(args.data_configs)
    else:
        data_configs = [_parse_args_to_config(args.data_args or [], "--data-arg")]

    # Resolve input path: data_factory if available, else --input files
    if data_fn is not None:
        if args.axes or args.data_axes:
            runner = BenchmarkMatrix(
                factory=factory,
                axes=pipeline_axes if args.axes else {"_": [{}]},
                data_factory=data_fn,
                data_axes=data_axes if args.data_axes else None,
                measurement=measurement,
            )
            if args.axes:
                print(runner.to_plan(), file=sys.stderr)
                print(file=sys.stderr)
            results = runner.run()
        else:
            results = BenchmarkSweep(
                factory=factory,
                configs=pipeline_configs,
                data_factory=data_fn,
                data_configs=data_configs,
                measurement=measurement,
            ).run()
    else:
        # Fall back to --input file paths (legacy input_fns path)
        if not args.input:
            raise CLIError(
                "no @data_factory found and no --input paths given. "
                "Either decorate a function with @data_factory, or pass --input path [...]."
            )
        input_fns, input_labels = _build_file_input_fns(args.input)
        if args.axes:
            runner = BenchmarkMatrix(
                factory=factory,
                axes=pipeline_axes,
                input_fns=input_fns,
                input_labels=input_labels,
                measurement=measurement,
            )
            print(runner.to_plan(), file=sys.stderr)
            print(file=sys.stderr)
            results = runner.run()
        else:
            results = BenchmarkSweep(
                factory=factory,
                configs=pipeline_configs,
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

    run_p = sub.add_parser(
        "run",
        help="Run a pipeline once without measurement",
        description="Discover @pipeline_factory and @data_factory and run the pipeline once.",
    )
    run_p.add_argument("pipeline_ref", metavar="MODULE[:FN]",
                       help="Pipeline factory: 'pkg.module' or 'pkg.module:fn_name'")
    run_p.add_argument("data_ref", metavar="DATA_MODULE[:FN]", nargs="?", default=None,
                       help="Data factory (auto-discovered in pipeline module if absent)")
    run_p.add_argument("--input", nargs="+", metavar="PATH",
                       help="Fallback input files when no @data_factory is present")
    run_p.add_argument("--arg", action="append", dest="args", metavar="KEY=VALUE",
                       help="Pipeline factory argument (repeatable), e.g. --arg slice_wh=480x480")
    run_p.add_argument("--data-arg", action="append", dest="data_args", metavar="KEY=VALUE",
                       help="Data factory argument (repeatable), e.g. --data-arg image_path=img.jpg")

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
                         help="Pipeline factory argument (repeatable), e.g. --arg max_concurrency=8")
    bench_p.add_argument("--data-arg", action="append", dest="data_args", metavar="KEY=VALUE",
                         help="Data factory argument (repeatable), e.g. --data-arg image_path=img.jpg")

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
    pipeline_group = sweep_p.add_mutually_exclusive_group()
    pipeline_group.add_argument("--arg", action="append", dest="args", metavar="KEY=VALUE",
                                help="Single pipeline config value (repeatable); mutually exclusive with --config/--axis")
    pipeline_group.add_argument("--config", action="append", dest="configs", metavar="JSON",
                                help="Explicit pipeline config as JSON dict (repeatable)")
    pipeline_group.add_argument("--axis", action="append", dest="axes",
                                metavar="KEY=V1,V2,...",
                                help="Pipeline axis for cartesian expansion (repeatable)")
    data_group = sweep_p.add_mutually_exclusive_group()
    data_group.add_argument("--data-arg", action="append", dest="data_args", metavar="KEY=VALUE",
                            help="Single data config value (repeatable); mutually exclusive with --data-config/--data-axis")
    data_group.add_argument("--data-config", action="append", dest="data_configs", metavar="JSON",
                            help="Explicit data config as JSON dict (repeatable)")
    data_group.add_argument("--data-axis", action="append", dest="data_axes",
                            metavar="KEY=V1,V2,...",
                            help="Data axis for cartesian expansion (repeatable)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            exit_code = cmd_run(args)
        elif args.command == "benchmark":
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
