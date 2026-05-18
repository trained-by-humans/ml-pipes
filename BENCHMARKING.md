# Benchmarking

ml-pipes includes a built-in benchmarking system that measures per-operator
latency across warmup + N measured runs and produces structured result objects
you can print, diff, save, and reload.

## Quick start

The lowest-level entry point is `Benchmark`. Construct it with a pipeline, an
input function, and a measurement config, then call `.run()`:

```python
from ml_pipes.benchmark import Benchmark, MeasurementConfig

result = Benchmark(
    pipeline=my_pipeline,
    input_fn=lambda: ("image.jpg", image_path, None, None),
    measurement=MeasurementConfig(runs=30, warmup=5),
    label="yolo8n-full",
    metadata={"model": "yolov8n.onnx"},
).run()

print(result.to_table())
```

`input_fn` is a zero-argument callable that returns
`(id: str, value: Any, tag: str | None, metadata: dict | None)`. The `tag`
and `metadata` fields are accepted but currently unused; pass `None` for both.

Output:

```
operator                    mean        p50        p95        p99     stddev        min        max
--------------------------------------------------------------------------------------------
total                      48.23      47.80      52.10      54.30       1.21      46.40      55.20
  Decode                    2.10       2.08       2.30       2.42       0.09       1.98       2.50
  Resize                    0.28       0.27       0.32       0.35       0.02       0.25       0.37
  Normalize                 2.22       2.20       2.40       2.51       0.08       2.10       2.60
  Infer                    39.19      38.90      43.10      45.20       1.10      37.80      46.00
  NMS                       0.21       0.20       0.25       0.27       0.01       0.19       0.28
  ToDetections              0.02       0.02       0.03       0.03       0.00       0.02       0.03
--------------------------------------------------------------------------------------------
runs: 30  (all values in ms)
```

## MeasurementConfig

```python
MeasurementConfig(
    runs=100,                           # measured runs (default: 100)
    warmup=10,                          # discarded warmup runs (default: 10)
    percentiles=(0.50, 0.95, 0.99),     # percentiles to compute (default: p50/p95/p99)
)
```

Warmup runs are always discarded before measurement begins — this gives ONNX
Runtime time to JIT-compile the graph and avoids cold-start I/O skewing results.

## BenchmarkResult

`Benchmark.run()` returns a `BenchmarkResult` with the following interface:

```python
result.label          # str — the label passed at construction
result.metadata       # dict — the metadata dict passed at construction
result.total          # InvocationStat — whole-pipeline latency
result.operators      # list[InvocationStat] — per-operator latency

result.to_table()                          # single-result operator table (str)
result.to_table(expand_regions=False)      # collapse Scatter/Gather child spans

BenchmarkResult.to_comparison_table(results)               # multi-column table
BenchmarkResult.to_comparison_table(results, expand_regions=False)

result.diff(other)     # BenchmarkDiff — delta table between two results
result.save("out.json")
result = BenchmarkResult.load("out.json")
result.slug(".json")   # filesystem-safe filename from the label
```

### Diff table

```python
diff = baseline.diff(candidate)
print(diff.to_table())
```

```
baseline : conf=0.10
candidate: conf=0.50
------------------------------------------------------------------
operator                        Δmean         Δmean%          Δp95          note
------------------------------------------------------------------
total                          +1.20ms        +2.5%         +1.80ms
  Decode                        0.00ms        +0.0%          0.00ms
  Infer                        +1.10ms        +2.8%         +1.60ms
  NMS                          +0.10ms        +4.8%         +0.20ms
------------------------------------------------------------------
```

Operators present in only one result show `only in baseline` or `only in candidate`
in the note column.

## BenchmarkBuilder

`BenchmarkBuilder` is a fluent builder for config sweeps. It expands combinations
automatically and returns a list of `BenchmarkResult` objects.

### Entry points

```python
BenchmarkBuilder.pipeline(p)   # start from a concrete Pipeline — no config sweep
BenchmarkBuilder.factory(f)    # start from a pipeline factory — enables config sweep
```

A concrete `Pipeline` is used as-is. A factory is any callable that accepts
`(config: dict)` and returns a `Pipeline`. The `@pipeline_factory` decorator
wraps a naturally-typed function into that shape (see below).

### Single run

```python
from ml_pipes.benchmark import BenchmarkBuilder

results = (
    BenchmarkBuilder.pipeline(my_pipeline)
    .data_input(lambda: ("img", image_path, None, None))
    .runs(30).warmup(5)
    .run()
)
```

### Config sweep with `pipeline_config_set`

Provide an explicit list of configs. Base args set via `pipeline_config()` are
merged into every entry — per-entry keys win:

```python
results = (
    BenchmarkBuilder.factory(my_pipeline_factory)
    .pipeline_config(model_path=model_path, output_path=output_path)   # base
    .pipeline_config_set([
        {"slice_wh": (320, 320)},
        {"slice_wh": (480, 480), "overlap_wh": (120, 120)},
    ])
    .data_input(input_fn)
    .runs(20).warmup(3)
    .run()
)
```

Effective configs are `{model_path: ..., output_path: ..., slice_wh: (320, 320)}`
and `{model_path: ..., output_path: ..., slice_wh: (480, 480), overlap_wh: (120, 120)}`.

### Axis sweep with `pipeline_config_axis`

Register one or more axes — the builder expands their cartesian product:

```python
results = (
    BenchmarkBuilder.factory(my_pipeline_factory)
    .pipeline_config(model_path=model_path, output_path=output_path)
    .pipeline_config_axis("slice_wh", (240, 240), (320, 320), (480, 480))
    .pipeline_config_axis("overlap_wh", (40, 40), (80, 80), (120, 120))
    .pipeline_config_filter(lambda c: c["overlap_wh"][0] < c["slice_wh"][0] // 2)
    .data_input(input_fn)
    .runs(20).warmup(3)
    .run()
)
```

`pipeline_config_set` and `pipeline_config_axis` are mutually exclusive.

### Data sweep

The data dimension is symmetric with the pipeline dimension:

```python
BenchmarkBuilder.factory(f)
    .data_factory(my_data_factory)
    .data_config_set([{"image": "a.jpg"}, {"image": "b.jpg"}])
    ...
```

Or with `data_inputs()` for a list of concrete `InputFn`s:

```python
BenchmarkBuilder.pipeline(my_pipeline)
    .data_inputs([fn_a, fn_b], labels=["coco", "voc"])
    ...
```

### Measurement and annotation

```python
.runs(30)
.warmup(5)
.percentiles(0.50, 0.90, 0.95, 0.99)

.label("my-experiment")       # prefix for result labels
.metadata({"git_sha": "abc"}) # merged into every result's metadata dict
```

### Previewing the sweep plan

Before running, inspect what will be executed:

```python
builder.plan()    # prints the config list or grid to stderr, returns str
builder.grid()    # 2D/3D ASCII grid of active vs filtered cells
```

```
plan: 9 combinations (6 active, 3 filtered)
  ○  slice_wh=(240, 240)  overlap_wh=(40, 40)
  ×  slice_wh=(240, 240)  overlap_wh=(80, 80)
  ...

grid: row=slice_wh  col=overlap_wh

              (40, 40)  (80, 80)  (120, 120)
(240, 240)       ○         ×          ×
(320, 320)       ○         ○          ×
(480, 480)       ○         ○          ○

○ = active  × = filtered
```

### Running

```python
results = builder.run()               # prints plan to stderr, returns list[BenchmarkResult]
results = builder.run(verbose=False)  # suppress plan output
```

## @pipeline_factory and @data_factory

These decorators let you write pipeline and data factories with natural Python
signatures. The builder calls them with a config dict — the decorator unpacks
it into keyword arguments:

```python
from ml_pipes import Pipeline, pipeline_factory, data_factory
from ml_pipes.benchmark import InputFn

@pipeline_factory
def my_pipeline(
    model_path: Path,
    slice_wh: tuple[int, int] = (320, 320),
    conf_threshold: float = 0.25,
) -> Pipeline:
    return Pipeline([...])


@data_factory
def my_data(image_path: Path = DEFAULT_IMAGE) -> InputFn:
    def fn():
        return (image_path.name, image_path, None, None)
    return fn
```

If the config dict contains an unknown key or is missing a required parameter,
the builder raises a descriptive `TypeError` before calling the factory:

```
TypeError: pipeline factory got unknown config key(s) ['typo_param']
           for config {'model_path': ..., 'typo_param': 1}

TypeError: pipeline factory is missing required config key(s) ['model_path']
           for config {}
```

If a factory returns the wrong type (`None` instead of a `Pipeline`, or a
non-callable instead of an `InputFn`), a descriptive error is raised
immediately rather than failing deep inside the measurement loop.

## CLI benchmarking

Modules that expose a `@pipeline_factory` and a `@data_factory` can be
benchmarked directly from the command line without writing a script:

```bash
# single run with default config
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli

# override one config key
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --arg slice_wh=480x480 --runs 20 --warmup 3

# explicit config sweep
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --config '{"slice_wh":[320,320]}' \
    --config '{"slice_wh":[480,480],"overlap_wh":[120,120]}' \
    --runs 20

# axis sweep — cartesian product
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --axis slice_wh=240x240,320x320,480x480 \
    --axis overlap_wh=40x40,80x80 \
    --runs 20
```

The CLI discovers the `@pipeline_factory` and `@data_factory` functions
automatically. If a module contains more than one of either, pass the fully
qualified name: `module:factory_fn`.

## Saving and reloading results

```python
result.save("results/yolo8n_run1.json")
result = BenchmarkResult.load("results/yolo8n_run1.json")

# filesystem-safe filename derived from the label
result.save(f"results/{result.slug('.json')}")
```

The JSON format is self-contained: label, metadata, total stats, and full
per-operator stats including percentiles and child spans. Results can be
forwarded to MLflow or W&B via `result.to_dict()`.

## Comparison table

`to_comparison_table` renders a side-by-side view across multiple results —
useful after a sweep:

```python
print(BenchmarkResult.to_comparison_table(results))
print(BenchmarkResult.to_comparison_table(results, expand_regions=False))
```

Percentile columns are the union of all results' percentile sets. A cell that
has no value for a given percentile (because that result was measured with
different percentiles) renders as `"-"`.

## Example scripts

| Script | What it demonstrates |
|---|---|
| `examples/benchmarks/run_yolo8_benchmark.py` | `Benchmark` directly: single run, config diff, structural diff |
| `examples/benchmarks/run_yolo8_benchmark_sweep.py` | `BenchmarkBuilder`: plain vs tiled pipeline side-by-side |
| `examples/benchmarks/run_yolo8_benchmark_sweep_axis.py` | Axis sweep: `slice_wh × overlap_wh` cartesian product with filter |
| `examples/benchmarks/run_yolo8_benchmark_variants.py` | Variant sweep: compare YOLOv8 n/s/m/l/x model sizes |
| `examples/benchmarks/run_yolo8_benchmark_cli.py` | CLI target: `@pipeline_factory` + `@data_factory` for `python -m ml_pipes benchmark` |

```bash
cd examples/benchmarks
python run_yolo8_benchmark.py --model n --runs 30
python run_yolo8_benchmark_sweep.py --model n --runs 20
python run_yolo8_benchmark_sweep_axis.py --model n --runs 20
python run_yolo8_benchmark_variants.py --variants n s --runs 20
```

## See also

- `TRACING.md` — the underlying per-operator tracing system that benchmarking builds on
- `PERFORMANCE.md` — throughput and batching guidance
