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
from ml_pipes.core import Operator


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
from ml_pipes.core import Pipeline

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

## Special Operator Shapes

Most operators are plain functions or `@Operator` classes with a normal
`__call__` boundary. A few superclass patterns exist because some operators
participate in framework mechanisms beyond a normal transform:

- Reach for the `ContextOp[In, Out]` superclass when the operator needs to
  read or write pipeline context while still presenting one explicit pipeline
  boundary.
- When the behavior cannot be expressed as one operator's transform and
  instead defines how a bounded group of operators should execute together,
  read [REGIONS.md](REGIONS.md). Region operators are implemented through the
  `RegionOpener[In, BodyIn]` and `RegionCloser[BodyOut, Out]` superclasses.
- Reach for the `SideEffectOp[T]` superclass when an operator should pass the
  value through unchanged while observing, logging, drawing, or saving.

These are still operators, but they participate in extra execution mechanics.
Read [ARCHITECTURE.md](ARCHITECTURE.md) for how context and regions fit into
pipeline execution, and [REGIONS.md](REGIONS.md) for region semantics and
examples.

## Features Built On Operators

The features below are not provided by the `@Operator` decorator itself.
They work because the framework treats operators as explicit pipeline
boundaries with meaningful input, output, and step identity.
When those boundaries are chosen well, composition, validation,
description, inspection, tracing, and benchmarking all become useful.

The examples below use the same tiny operator chain:

```python
from ml_pipes.collectors import PrintCollector
from ml_pipes.core import Operator, Pipeline


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
> That handoff between operators is the base layer the rest of the framework
> builds on.

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
> Validation is only as good as the boundaries it can read, so precise
> `__call__` annotations matter.

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
> Good operator names and readable constructor arguments make pipeline
> descriptions easier to scan.

### Inspection

Inspection captures the value flowing through each operator boundary.

```python
from ml_pipes.inspection import PipelineInspector, TextBlock

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
> Inspection is easier to read when each operator marks one clear step instead
> of hiding several steps inside one fused block.

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
> Tracing is most useful when each timing line corresponds to one meaningful
> step rather than a mixed block of work.

### Benchmark

Benchmarking repeats the pipeline over many runs and aggregates latency at the
same operator boundaries tracing uses.

```python
from ml_pipes.benchmark import Benchmark, MeasurementConfig

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
> Benchmarking reuses those same operator boundaries, but compares them across
> repeated runs instead of one execution.

## Design Principles

Operators should satisfy a few hard constraints so they stay safe to compose,
swap, and reason about.

**Atomic.** One meaningful pipeline boundary, not a whole workflow.

**Stateless.** Keep only construction-time config (`name`, `axis`,
`threshold`, `prefix`, etc.), never hidden runtime memory.

**Effect-explicit.** If an operator touches context, opens a region, performs a
side effect, or crosses a runtime boundary, that should be obvious in its type
and behavior.

**Reusable at the right layer.** Put broadly reusable behavior in shared
operators; keep project glue and experiments in local pipeline code.

**Composable.** Operators should be easy to insert, remove, reorder, validate,
inspect, trace, and benchmark as standalone boundaries.

## Best Practices For Creating Operators

### Scope

- Each operator should do one meaningful transformation or control-flow step.
  If the behavior is bigger than that, compose multiple operators instead of
  fusing a whole workflow into one.
- Break operators where you want the pipeline boundary to be. If you want
  validation, inspection, tracing, or failures to stop at a specific point,
  make that point its own operator instead of fusing past it.
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
- Use a fixed-length tuple when the operator returns multiple values that are
  small, short-lived, and positionally obvious to the next few steps. This
  pairs naturally with the multi-input pattern described in the input rules
  above.
- Remember that a fixed-length tuple is structural in `ml-pipes`. Returning
  `tuple[A, B]` publishes two positional pipeline values, and tuple-routing
  operators such as `Pick` and `Recall` will act on that structure. If the
  tuple is semantically one payload, wrap it in a dedicated object instead of
  returning the raw tuple.
- Use a dedicated dataclass when fields have high cohesion, represent one
  named result, and are usually used together by name.
- A useful pattern is the carry-forward tuple: return the original value
  together with a new derived value when later pipeline steps are expected to
  keep using both. Rendering is a good example: an operator can return
  `(rendered_image, detections)` so later steps can save the image while still
  rendering, filtering, or logging the detections in different ways.
- In-place mutation is acceptable when the payload is already mutation-oriented
  and that keeps the pipeline clearer.
- Use a registry-style payload when many operators cooperatively read and write
  named intermediate slots, as with `TensorRegistry` or `TorchTensorRegistry`.

### Naming And Aliases

- Choose one primary name for each operator surface based on what the operator
  does.
- Do not name an operator relative to other operators or to one specific
  pipeline shape. Operators can appear in any position and in many different
  pipelines.
- Use aliases when a domain-specific spelling makes a common use case read
  better.
- Treat aliases as alternate spellings of the same operator, not as separate
  operators with different semantics.

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

### Special Operators

- Reach for the `SideEffectOp[T]` superclass when the operator should behave
  like `T -> T` and its semantic job is to observe, log, draw, or save rather
  than transform the flowing value.
- Reach for the `ContextOp[In, Out]` superclass when the operator genuinely
  needs to read from or write to pipeline context as part of its behavior.
  Do not extend `ContextOp` just to read one specific key from context. If the
  operator has a specific dependency, define it in the operator signature and
  let pipeline composition provide it explicitly.
- Reach for the region mechanism (explained in [REGIONS.md](REGIONS.md)) when
  the behavior cannot be expressed as one operator's transform and instead
  defines how a bounded group of operators should execute together.

### Errors And Verification

- Error messages should be direct and local to the operator that detects the
  problem.
- Verify the operator inside a real pipeline. `validate()` checks its
  boundary, `inspect()` shows what value flows through it, and tracing or
  benchmarking can be added once correctness is already established.

## Operator Packages

So far this page has focused on what operators are and how to create them.
In practice, many reusable operators already exist, so most users should start
from [PACKAGES.md](PACKAGES.md) and then check the linked package index for
the operators they need.

Start from `ml_pipes.standard` for shared generic building blocks such as
routing, context, regions, and data-preparation work.

Example: if postprocess is still tensor-shaped, start from the tensor package
and its index in
[packages/tensor/docs/INDEX.md](../packages/tensor/docs/INDEX.md).
