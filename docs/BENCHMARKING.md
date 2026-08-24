# Benchmarking

Tracing (see [TRACING.md](TRACING.md)) gives you a per-operator breakdown of a
single pipeline call. Benchmarking builds on top of tracing to give you the
statistical spread across many: mean, percentiles, and stddev per operator,
collected over a controlled warmup + measurement loop and returned as a
structured result you can print, diff, save, and reload.

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
  0:Decode                  2.10       2.08       2.30       2.42       0.09       1.98       2.50
  1:Resize                  0.28       0.27       0.32       0.35       0.02       0.25       0.37
  2:Normalize               2.22       2.20       2.40       2.51       0.08       2.10       2.60
  3:Infer                  39.19      38.90      43.10      45.20       1.10      37.80      46.00
  4:NMS                     0.21       0.20       0.25       0.27       0.01       0.19       0.28
--------------------------------------------------------------------------------------------
runs: 30  (all values in ms)
```

## Measurement configuration

`MeasurementConfig` controls the measurement loop:

```python
MeasurementConfig(
    runs=100,                           # measured runs (default: 100)
    warmup=10,                          # discarded warmup runs (default: 10)
    percentiles=(0.50, 0.95, 0.99),     # percentiles to compute (default: p50/p95/p99)
)
```

Warmup runs are always discarded before measurement begins — this allows the
runtime to JIT-compile, fill caches, and reach steady state before samples are
recorded.

## Results

`Benchmark.run()` returns a `BenchmarkResult`:

```python
result.label          # str — the label passed at construction
result.metadata       # dict — the metadata dict passed at construction
result.total          # InvocationStat — whole-pipeline latency
result.operators      # list[InvocationStat] — per-operator latency
```

### Single-result table

```python
result.to_table()                          # per-operator latency table (str)
result.to_table(expand_regions=False)      # collapse child spans for any region operator
```

The output format is shown in the quick start above.

### Diff table

Compare two results operator by operator — useful for measuring the impact of
a config change or a structural pipeline difference:

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
  0:Decode                      0.00ms        +0.0%          0.00ms
  1:Resize                      0.00ms        +0.0%          0.00ms
  2:Normalize                   0.00ms        +0.0%          0.00ms
  3:Infer                      +1.10ms        +2.8%         +1.60ms
  4:NMS                        +0.10ms        +4.8%         +0.20ms
------------------------------------------------------------------
```

Operators present in only one result show `only in baseline` or `only in
candidate` in the note column.

### Comparison table

`BenchmarkResult.to_comparison_table` renders a side-by-side view across
multiple results — useful after a sweep to compare all configurations at once:

```python
print(BenchmarkResult.to_comparison_table(results))
print(BenchmarkResult.to_comparison_table(results, expand_regions=False))
```

Percentile columns are the union of all results' percentile sets. A cell that
has no value for a given percentile renders as `"-"`.

### Saving and reloading

Results are serializable to JSON and can be reloaded in a later session:

```python
result.save("results/yolo8n_run1.json")
result = BenchmarkResult.load("results/yolo8n_run1.json")

# filesystem-safe filename derived from the label
result.save(f"results/{result.slug('.json')}")
```

The JSON format is self-contained: label, metadata, total stats, and full
per-operator stats including percentiles and child spans. Results can be
forwarded to MLflow or W&B via `result.to_dict()`.

## Sweeping

`Benchmark` measures one pipeline against one input. When you want to compare
multiple pipeline configurations — different model sizes, tile sizes, confidence
thresholds — or run the same pipeline against multiple datasets, you need a
sweep.

A sweep has two independent dimensions:

- **Pipeline dimension** — vary the pipeline itself by providing a factory and
  a set of configs. The factory is called once per config to construct each
  pipeline variant.
- **Data dimension** — vary the input by providing a data factory and a set of
  data configs. The data factory is called once per data config to produce the
  input function for that variant.

When both dimensions are provided, every pipeline config is run against every
data config. With 3 pipeline configs and 2 data configs, that is 6 results.

`BenchmarkBuilder` is the fluent API for building sweeps. It expands
combinations automatically and returns one `BenchmarkResult` per combination.

### Measurement and annotation

The same measurement options from `MeasurementConfig` are available on the
builder:

```python
builder
    .runs(30)
    .warmup(5)
    .percentiles(0.50, 0.90, 0.95, 0.99)
```

You can also annotate all results in a sweep with a shared label prefix and
metadata dict — useful for tagging an experiment or capturing environment info:

```python
builder
    .label("experiment-v2")           # prefix for all result labels in this sweep
    .metadata({"git_sha": "abc123"})  # merged into every result's metadata dict
```

### Single run

When neither a pipeline factory nor a data factory is involved, there is
nothing to sweep over — the builder measures a single fixed pipeline against a
single fixed input and returns one result. This is equivalent to using
`Benchmark` directly:

```python
from ml_pipes.benchmark import BenchmarkBuilder

results = (
    BenchmarkBuilder.pipeline(my_pipeline)
    .data_input(lambda: ("img", image_path, None, None))
    .runs(30).warmup(5)
    .run()
)
```

### Pipeline configuration sweep

To sweep over pipeline configurations, start from a factory rather than a
concrete pipeline. The builder calls the factory once per config combination
to construct each pipeline variant.

#### Pipeline factory

Use `@pipeline_factory` for reusable declared factories. Parameters without
defaults are required config keys; parameters with defaults are optional:

```python
@pipeline_factory
def my_pipeline(model_path: Path, conf_threshold: float = 0.25) -> Pipeline:
    return Pipeline([Infer(model_path), NMS(conf_threshold=conf_threshold), ...])

BenchmarkBuilder.factory(my_pipeline)
```

Place `@pipeline_factory` on the top decorator line. The exported symbol must
itself be the `PipelineFactory`; wrapping it afterward is unsupported.

For ad hoc cases, pass a plain callable directly to
`BenchmarkBuilder.factory(...)`. It should accept config as keyword arguments:

```python
factory = lambda model_path, conf_threshold=0.25: (
    decode() + yolo8_inference_pipeline(model_path, conf_threshold=conf_threshold)
)

results = (
    BenchmarkBuilder.factory(factory)
    .pipeline_config(model_path=model_path)
    .run()
)
```

The decorator returns a factory object that stays directly callable while also
exposing `build(...)` for discovery and config-driven execution.

Unknown config keys, missing required parameters, and wrong return types all
produce descriptive errors that name the offending key and config:

```
TypeError: pipeline factory got unknown config key(s) ['typo_param']
           for config {'model_path': ..., 'typo_param': 1}
```

#### Handpicked configs with `pipeline_config_set`

Provide an explicit list of configs to run. Base args set via
`pipeline_config()` are merged into every entry — per-entry keys win:

```python
results = (
    BenchmarkBuilder.factory(my_pipeline)
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

Effective configs:

```python
{"model_path": model_path, "output_path": output_path, "slice_wh": (320, 320)}
{"model_path": model_path, "output_path": output_path, "slice_wh": (480, 480), "overlap_wh": (120, 120)}
```

#### Axis sweep with `pipeline_config_axis`

Register one or more axes — the builder expands their cartesian product. Use
`pipeline_config_filter` to drop combinations that are not meaningful:

```python
results = (
    BenchmarkBuilder.factory(my_pipeline)
    .pipeline_config(model_path=model_path, output_path=output_path)
    .pipeline_config_axis("slice_wh", (240, 240), (320, 320), (480, 480))
    .pipeline_config_axis("overlap_wh", (40, 40), (80, 80), (120, 120))
    .pipeline_config_filter(lambda c: c["overlap_wh"][0] < c["slice_wh"][0] // 2)
    .data_input(input_fn)
    .runs(20).warmup(3)
    .run()
)
```

> [!NOTE]
> `pipeline_config_set()` and `pipeline_config_axis()` are mutually exclusive.

### Data sweep

The data dimension is symmetric with the pipeline dimension. Provide a data
factory and a set of data configs to run the same pipeline against multiple
inputs.

#### Data factory

Use `@data_factory` for reusable declared data factories:

```python
@data_factory
def my_data(image_path: Path) -> InputFn:
    def fn():
        return (image_path.name, image_path, None, None)
    return fn

BenchmarkBuilder.factory(my_pipeline).data_factory(my_data)
```

Place `@data_factory` on the top decorator line. The exported symbol must
itself be the `DataFactory`; wrapping it afterward is unsupported.

For ad hoc cases, pass a plain callable directly to `.data_factory(...)`. It
should accept config as keyword arguments:

```python
results = (
    BenchmarkBuilder.factory(my_pipeline)
    .data_factory(lambda image_path: lambda: (image_path.name, image_path, None, None))
    .data_config(image_path=Path("coco_val.jpg"))
    .run()
)
```

As with `@pipeline_factory`, the decorator returns a factory object that stays
directly callable while exposing `build(...)` for config-driven use.

The same error feedback applies as for the pipeline factory.

#### Handpicked data configs with `data_config_set`

```python
results = (
    BenchmarkBuilder.factory(my_pipeline)
    .pipeline_config(model_path=model_path)
    .data_factory(my_data)
    .data_config_set([
        {"image_path": Path("coco_val.jpg")},
        {"image_path": Path("voc_sample.jpg")},
    ])
    .run()
)
```

#### Axis sweep with `data_config_axis`

```python
results = (
    BenchmarkBuilder.factory(my_pipeline)
    .pipeline_config(model_path=model_path)
    .data_factory(my_data)
    .data_config_axis("image_path", Path("a.jpg"), Path("b.jpg"), Path("c.jpg"))
    .run()
)
```

> [!NOTE]
> `data_config_set()` and `data_config_axis()` are mutually exclusive.

### Previewing the sweep plan

Before running, `plan()` previews the pipeline config dimension — the list of
configs that will be used to construct pipeline variants, including which axis
combinations are active or filtered out. Data configs are resolved separately
and are not shown. For axis sweeps with 2 or 3 axes, a 2D/3D grid is included:

```python
builder.plan()    # prints the config list or grid to stderr, returns str
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

### Running the sweep

```python
results = builder.run()               # prints plan to stderr, returns list[BenchmarkResult]
results = builder.run(verbose=False)  # suppress plan output
```

### Sweep results

`run()` returns a flat `list[BenchmarkResult]` — one per combination. Each
result can be printed individually or passed to `to_comparison_table` for a
side-by-side view across all combinations:

```python
results = builder.run()

# inspect one result
print(results[0].to_table())

# compare all results side by side
print(BenchmarkResult.to_comparison_table(results))
print(BenchmarkResult.to_comparison_table(results, expand_regions=False))
```

Each result also carries the pipeline and data configs that produced it in its
`metadata` dict, so you can trace any cell back to its exact configuration:

```python
for result in results:
    print(result.label, result.metadata["pipeline_config"])
```

## CLI benchmarking

Any module that exposes a `@pipeline_factory` can be benchmarked directly from
the command line using `python -m ml_pipes benchmark`, without writing a
benchmark script. The CLI builds and runs a `BenchmarkBuilder` sweep under the
hood — the same config, axis, and filter semantics apply.

For the data side, decorate a function in the module with `@data_factory` to
get the full config-driven sweep behaviour. If no `@data_factory` is present,
you can pass one or more file paths directly via `--input` as a quick
alternative for simple file-based inputs.

The CLI discovers the `@pipeline_factory` and `@data_factory` functions
automatically by scanning the module. Keep those decorators on the top
decorator line so the exported symbol is the factory itself. If a module
contains more than one of either, pass the fully qualified name:
`module:factory_fn`.

```bash
# single run, input passed directly
python -m ml_pipes benchmark my_module --input image.jpg

# single run with default config from @data_factory
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli

# override one config key
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --arg slice_wh=480x480

# explicit config sweep
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --config '{"slice_wh":[320,320]}' \
    --config '{"slice_wh":[480,480],"overlap_wh":[120,120]}'

# axis sweep — cartesian product
python -m ml_pipes benchmark examples.benchmarks.run_yolo8_benchmark_cli \
    --axis slice_wh=240x240,320x320,480x480 \
    --axis overlap_wh=40x40,80x80
```

## Example scripts

| Script | What it demonstrates |
|---|---|
| `examples/benchmarks/run_yolo8_benchmark.py` | `Benchmark` directly: one measured run plus one structural diff |
| `examples/benchmarks/run_yolo8_benchmark_sweep.py` | `BenchmarkBuilder`: one plain baseline plus a small tiled `slice_wh` sweep |
| `examples/benchmarks/run_yolo8_benchmark_variants.py` | Variant sweep: compare YOLOv8 n/s/m/l/x model sizes |
| `examples/benchmarks/run_yolo8_benchmark_cli.py` | CLI target: `@pipeline_factory` + `@data_factory` for `python -m ml_pipes benchmark` |

```bash
cd examples/benchmarks
python run_yolo8_benchmark.py --runs 30
python run_yolo8_benchmark_sweep.py --runs 20
python run_yolo8_benchmark_variants.py --variants n s --runs 20
```

## See also

- `TRACING.md` — the underlying per-operator tracing system that benchmarking builds on
- `PERFORMANCE.md` — throughput and batching guidance
