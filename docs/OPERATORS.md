# Operators

## What Operators Are

Semantically, operators are the steps a pipeline takes to turn an input into
an output.

The pipeline itself does not have any task or workflow specific logic. The pipeline is the
framework that runs operators in order, threads context, and layers tooling on
top of those steps. 
If a pipeline converts an image into detections, or a text string into a list
of tokens, the interesting logic lives in the operators:

- decode the input
- normalize it
- run inference
- extract tensors
- project coordinates
- convert to the final output object

In that sense, a pipeline is closer to a harness for
operators than to the operators themselves.

## How Operators Look In Code

Concretely, an operator is anything the pipeline can execute as one step.

For simple local logic, a plain function is enough:

```python
def strip_text(text: str) -> str:
    return text.strip()
```

For reusable configured logic, use a class with `__call__`. Decorating it with
`@Operator` makes it self-describing in `repr()` and `describe()`:

```python
from ml_pipes import Operator


@Operator
class Prefix:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, text: str) -> str:
        return f"{self.prefix}{text}"
```

```python
repr(Prefix("tag: "))
# Prefix('tag: ')
```

Pipelines can mix both forms freely:

```python
from ml_pipes import Pipeline

pipeline = Pipeline([strip_text, Prefix("tag: ")])
assert pipeline("  hello  ") == "tag: hello"
```

## What Operators Are Made Of

Most reusable operators have four parts the pipeline cares about:

```python
@Operator
class Prefix:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def __call__(self, text: str) -> str:
        return f"{self.prefix}{text}"
```

- **Input** is the `__call__` parameter annotation. Here, the operator accepts
  `str`.
- **Output** is the `__call__` return annotation. Here, the operator produces
  `str`.
- **Config** is the constructor state captured when you build the operator.
  Here, that is `prefix`.
- **Static validation rule** is the boundary the validator reads from the
  operator definition. In the simple case, `__call__` annotations are enough:
  the validator can see that this operator accepts `str` and returns `str`.

When that static shape is not enough, define `resolve_contract(...)`. Use it
for operators whose real boundary depends on the current upstream type, stored
context, or tuple routing. Most normal transform operators do not need it.

## Features Built On Operators

The pipeline's tooling works because operators define explicit boundaries.

The examples below use the same tiny operator chain:

```python
from ml_pipes import Operator, Pipeline, PrintCollector


def strip_text(text: str) -> str:
    return text.strip()


@Operator
class Lowercase:
    def __call__(self, text: str) -> str:
        return text.lower()


@Operator
class SplitWords:
    def __init__(self, delimiter: str) -> None:
        self.delimiter = delimiter

    def __call__(self, text: str) -> list[str]:
        return text.split(self.delimiter)
```

### Composition

Composition is the core behavior: the output of one operator becomes the input
to the next operator. The pipeline does not need task-specific logic here; it
just runs one operator, takes the returned value, and feeds it to the next.

```python
pipeline = Pipeline([strip_text, Lowercase(), SplitWords(delimiter=" ")])
pipeline("  Hello World  ")
# ['hello', 'world']
```

> [!TIP]
> That chaining behavior is the foundation the rest of the framework builds on.

### Validation

Validation checks whether one operator's output boundary is compatible with the
next operator's input boundary.

```python
pipeline = Pipeline([strip_text, Lowercase(), SplitWords(delimiter=" ")])
contract = pipeline.validate(strict=True)
print(f"Input: {contract.input_type}, Output: {contract.output_type}")
```

```text
Input: <class 'str'>, Output: list[str]
```

> [!TIP]
> This is why reusable operators should carry precise `__call__` annotations:
> the validator reasons about the pipeline through operator boundaries.

### Description

Because a pipeline is a list of operators, the operator chain is directly
describable.

```python
pipeline = Pipeline([strip_text, Lowercase(), SplitWords(delimiter=" ")])
pipeline.describe()
```

```text
Pipeline([
  strip_text,
  Lowercase(),
  SplitWords(delimiter=' '),
])
```

> [!TIP]
> This is why reusable operators should have meaningful names and, when
> configured, meaningful constructor arguments.

### Inspection

Inspection captures the value flowing through each operator boundary.

```python
from ml_pipes import PipelineInspector, TextBlock

pipeline = Pipeline([strip_text, Lowercase(), SplitWords(delimiter=" ")])
result = pipeline.inspect("  Hello World  ")
inspector = PipelineInspector().register_output_formatter(
    str,
    lambda value: [TextBlock("", [("", value)])],
)

views = inspector.build_views(result)
label_width = max(len(view.label) for view in views)

for view in views:
    cell = view.blocks[0]
    text = " | ".join(f"{key} {value}".strip() for key, value in cell.rows)
    print(f"{view.label:<{label_width}} : {text}")
```

```text
0:strip_text : Hello World
1:Lowercase  : hello world
2:SplitWords : [0] hello | [1] world
```

> [!TIP]
> This is one reason small single-purpose operators are easier to debug than
> large fused blocks.

### Tracing

Tracing reports latency per operator step.

```python
pipeline = Pipeline([strip_text, Lowercase(), SplitWords(delimiter=" ")])
pipeline.set_tracing(PrintCollector())
pipeline("  Hello World  ")
pipeline.set_tracing(None)
```

```text
  0:strip_text                      0.01ms  (21.6%)
  1:Lowercase                       0.01ms  (43.1%)
  2:SplitWords                      0.01ms  (17.4%)
  total                             0.03ms
```

> [!TIP]
> If an operator is too broad or mixes unrelated work, tracing becomes less
> useful because the latency is no longer attributed to a meaningful boundary.

### Benchmark

Benchmarking repeats the pipeline over many runs and aggregates latency at the
same operator boundaries tracing uses.

```python
from ml_pipes import Benchmark, MeasurementConfig

result = Benchmark(
    pipeline,
    input_fn=lambda: ("sample", "  Hello World  ", None, None),
    measurement=MeasurementConfig(runs=5, warmup=1, percentiles=(0.50,)),
).run()

print(result.to_table())
```

```text
operator            mean        p50     stddev        min        max
--------------------------------------------------------------------
total              0.04       0.03       0.00       0.03       0.04
0:strip_text       0.01       0.01       0.00       0.01       0.01
1:Lowercase        0.01       0.01       0.00       0.01       0.02
2:SplitWords       0.01       0.01       0.00       0.01       0.01
--------------------------------------------------------------------
runs: 5  (all values in ms)
```

> [!TIP]
> Benchmarking is most useful after the pipeline already works: it keeps the
> same operator boundaries as tracing, but measures them across repeated runs.

## Design Principles

Every operator in the library is designed to uphold the following properties.
They are not style guidelines — they are what makes operators safe to compose
and swap without side effects.

**Atomically meaningful.** Each operator should represent one meaningful
boundary in the pipeline: normalize text, store a value in context, open a
scatter region, draw boxes, or convert one box format to another. Larger
workflows should be built by composing operators, not by fusing many concerns
into one broad step.

**Stateless.** An operator should hold only stable configuration given at
construction time (`name`, `axis`, `threshold`, `prefix`, etc.). It should not
accumulate hidden runtime state across calls. This is primarily an execution
safety property: stateless operators are easier to run in parallel or
distributed pipelines, easier to scale, and easier to debug because the same
input under the same config produces the same result.

**Effect-explicit.** If an operator touches context, opens a region, performs
a side effect, or crosses a runtime boundary, that role should be explicit in
its type and behavior. Control flow and side effects should be first-class
operator semantics, not hidden inside an otherwise generic transform.

**Reusable at the right layer.** Shared operators should capture behavior that
is genuinely reusable across pipelines. Project-specific assumptions, one-off
workflow glue, and temporary experiments belong in local pipeline code, not in
the shared operator surface.

**Composable.** An operator should be easy to insert, remove, reorder,
validate, inspect, trace, and benchmark as a standalone boundary. The clearer
and smaller the operator, the more useful the surrounding pipeline tooling
becomes.

## Best Practices For Creating Operators

### Scope

- Each operator should do one meaningful transformation or control-flow step.
  If the behavior is bigger than that, compose multiple operators instead of
  fusing a whole workflow into one.
- Start with the simplest form that fits. Use a plain function for short local
  transforms. Move to a class with `@Operator` when the logic needs
  configuration, reuse, or clearer description output.

### Config

- Operator config is a pipeline-build-time value. Put it in `__init__`, and
  normalize and validate it once at construction time.
- Config is static. If the same input under the same config produces different
  output over time because the operator remembered previous calls, that is
  hidden state, not config.

### Runtime Boundary

- Use the invocation value for data that can change from one pipeline call to
  the next without rebuilding the pipeline.
- Do not pass operator config as per-invocation payload. If a value is really
  part of how the operator is configured, put it in `__init__`.
- Declare accurate `__call__` annotations so the operator's runtime boundary
  is explicit.
- Use generics and `TypeVar` when the operator preserves or transforms the
  relationship between input and output types.
- Avoid `Any` in `__call__` annotations unless the boundary is genuinely
  impossible to express more precisely.

### Input

- Define an input boundary that makes sense for the operator's actual job.
- Avoid passing a large payload to an operator when it only needs one small
  part of it. Use `Pick` or `Select` to pass only what the operator needs,
  which usually makes the operator more reusable.
- When an operator truly consumes multiple separate values, prefer multiple
  positional inputs over wrapping everything into one input object. The
  pipeline can unpack fixed-length tuple values automatically, which usually
  makes the operator boundary easier to read.

### Output

- Return a single value when the operator produces one semantic result.
- For ordinary transforms, treat the input as immutable and return a new value
  instead of mutating the input in place.
- Use a fixed-length tuple when the operator returns multiple values. This
  pairs naturally with the multi-input pattern described in the input rules
  above.
- Use a dedicated dataclass when fields have high cohesion, represent one
  named result, and are usually used together.
- A useful pattern is the carry-forward tuple: return the original value
  together with a new derived value when later pipeline steps are expected to
  keep using both. Rendering is a good example: an operator can return
  `(rendered_image, detections)` so later steps can save the image while still
  rendering, filtering, or logging the detections in different ways.
- In-place mutation is acceptable when the payload is already mutation-oriented
  and that keeps the pipeline clearer, as with registry-style payloads.

### Static Verification

- Define `resolve_contract(...)` only when normal annotations cannot express
  the real boundary precisely, such as upstream-type-dependent, context-aware,
  or tuple-routing operators.
- Put only boundary logic inside `resolve_contract(...)`. Checks like simple
  config validation belong in `__init__`; use `resolve_contract(...)` only for
  checks that depend on boundary information.
- Use the upstream type effectively to compute an accurate output type. If the
  operator preserves or transforms part of the incoming shape, reflect that in
  the returned contract instead of discarding it.
- Return the narrowest contract you can justify. Do not fall back to `Any`
  unless the boundary is genuinely impossible to express more precisely.
- Keep `resolve_contract(...)` aligned with runtime behavior. If `__call__` or
  `apply()` changes what the operator actually accepts or returns,
  `resolve_contract(...)` should change with it.

### Effects And Specialized Operators

- Side effects should be explicit. Use `SideEffectOp` when an operator's
  semantic job is to observe, log, draw, or save while passing the value
  through unchanged.
- Use `ContextOp` only when the operator genuinely reads or writes the
  side-channel context.
- Region behavior should also be explicit. Opening or closing a special
  execution region should be visible in the operator type and pipeline shape.

### Errors And Verification

- Error messages should be direct and local to the operator that detects the
  problem.
- Verify the operator inside a real pipeline. `validate()` checks its
  boundary, `inspect()` shows what value flows through it, and tracing or
  benchmarking can be added once correctness is already established.
