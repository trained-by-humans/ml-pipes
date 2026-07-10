# Pipeline Composition

ml-pipes follows a bring-your-own-code model. A pipeline can be built from
plain Python callables, your own operators, or operator packages.
The point is not to force everything through a package surface. The point is
to give your steps explicit execution boundaries so they can be composed,
validated, inspected, traced, and benchmarked.

## What A Pipeline Is

Semantically, a pipeline is a sequence of explicit steps that turns one input
boundary into one output boundary.

Concretely, you create one by passing an ordered list of steps to
`Pipeline([...])`. Each step receives the current value and returns the next
one.

The pipeline itself is the execution harness around those steps. The
task-specific behavior still lives in your operators and callables.

The goal of composition is not to hide a whole workflow inside one operator.
The goal is to choose boundaries that make the workflow understandable,
correct, and observable.

## Composing Pipelines From Existing Code

### Start From The Goal

When building a pipeline from code, start from the outcome you want:

1. define the pipeline input
2. define the pipeline output
3. identify the intermediate values that matter
4. turn each meaningful transformation into a separate step

Those steps often line up with places where:

- the value changes shape or type
- you want to inspect the intermediate result
- you want to isolate a side effect or runtime boundary
- you want to measure latency separately

If one step is hard to name, inspect, or measure, it is often too large.

### Use Validation, Inspection, Tracing, And Benchmarking

The pipeline tooling helps you shape the composition itself:

- `validate()` checks whether the boundaries between steps are compatible
- `inspect()` shows what value each step actually produced
- tracing shows per-step latency for one run
- benchmarking repeats the same pipeline across many runs to measure steady
  performance

That means composition is not just about execution order. It is also about
choosing steps that are easy to validate, debug, and profile.

### Build In Observable Steps

In practice, a good composition loop looks like this:

1. start with the smallest pipeline that proves the end-to-end path
2. validate the boundary contract
3. inspect a representative input
4. split broad steps into smaller ones where you need more visibility
5. trace or benchmark only after the pipeline already works

Example: starting from a script like `prepare_dataset.py`, the goal might be
to take raw rows in and get cleaned unique records out.

You do not need to implement every step up front. Start by naming the
boundaries you want to observe, then fill them in:

```python
from ml_pipes.core import Operator, Pipeline


@Operator
class NormalizeWhitespace:
    def __call__(self, text: str) -> str:
        return " ".join(text.split())


pipeline = Pipeline([
    LoadRows(),
    ExtractMessageText(),
    NormalizeWhitespace(),
    DropRowsWithUrl(),
    DeduplicateMessages(),
    WriteDataset("clean.jsonl"),
])
```

> [!TIP]
> The important point is not the specific operators. It is that the pipeline is
> broken into steps where correctness and execution can be observed directly.

## Composing With Existing Operators

When existing operators already cover part of the problem, use them first. This usually gives you:

- clearer boundaries
- better validation
- better inspection and tracing output
- less one-off code to maintain

The normal composition order is:

1. start from existing operators
2. compose as much of the pipeline as possible from them
3. add new operators only where the existing operator surface does not cover
   the need

### Start From The Closest Operator Package

Start from the shared building blocks, then move outward only as needed:

- begin with `ml_pipes.standard`, which provides the generic building blocks
  for data manipulation, routing, context, regions, and other reusable steps
  that shape composition and performance
- then check the package index in [PACKAGES.md](PACKAGES.md) to see whether a
  package already exists for the current domain or task boundary; for example,
  tensor-shaped postprocess should usually start from the `ml_pipes.tensor` package
- if you still cannot find the right package surface, check the runnable
  examples in [../examples/README.md](../examples/README.md) to see whether a
  similar pipeline already exists before adding new local operators

### Add New Operators Only Where Needed

When you do need a new operator, keep it narrow and follow
[OPERATORS.md](OPERATORS.md).

Examples in this repo:

- [../examples/run_yolo8_onnx.py](../examples/run_yolo8_onnx.py) starts from
  existing vision and inference operators
- [../examples/run_sms_spam_prepare.py](../examples/run_sms_spam_prepare.py)
  shows composition around data preparation and cleanup where local operators
  still make sense

## Composition Best Practices

Composition quality still depends on operator quality, so the operator best
practices in [OPERATORS.md](OPERATORS.md) apply here as well.

The points below focus on how multiple steps work together inside one
pipeline.

- Use `Store` / `Recall` when one derived value needs to be recovered later
  after several unrelated steps. This keeps the main flowing boundary focused
  on the value the next operators actually work on.
- Use a registry-style workspace when one stage needs to accumulate many named
  intermediates and later steps read them by name rather than by tuple
  position. `TensorRegistry` is one example of this pattern, but the same idea
  also appears in other runtime-oriented systems that work with named tensors
  or buffers.

## Combining Existing Pipelines

When you already have named pipelines, ml-pipes gives you two ways to combine
them: **join** and **merge**.

Use this section when the question is no longer "how do I build one pipeline
from steps?" but "how do I combine pipelines that already exist?"

### Join

Joining two pipelines keeps the original pipelines as distinct,
self-contained blocks. The joined pipeline treats each block as one step.

Two consequences follow directly:

- **Isolated context**: each block runs with a fresh `Context`. Internal
  `Store`/`Recall` keys are invisible outside the block, and outer keys are
  invisible inside.
- **Live reference**: the joined pipeline holds a reference to the original
  pipeline object. Mutating the source after composition is reflected at
  runtime.

```
Before

 Pipeline A                Pipeline B
┌──────────────────┐      ┌──────────────────┐
│ Op1 → Op2 → Op3  │  >>  │ Op4 → Op5 → Op6  │
└──────────────────┘      └──────────────────┘

After

 Pipeline (A >> B)
┌───────────────────────────────────────────────┐
│  ┌──────────────────┐   ┌──────────────────┐  │
│  │ Op1 → Op2 → Op3  │ → │ Op4 → Op5 → Op6  │  │
│  └──────────────────┘   └──────────────────┘  │
└───────────────────────────────────────────────┘
```

Join is the right choice when you want to keep boundaries between whole
pipeline blocks.

#### Inside a pipeline definition: `embed(p)` / `Embed(p)`

Use these inside `Pipeline([...])` to join another named pipeline.

```python
preprocess = Pipeline([Resize((640, 640)), Normalize()])
infer = Pipeline([Infer("model.onnx"), Extract("boxes", "scores", "classes")])

detection = Pipeline([
    Decode(),
    embed(preprocess),
    embed(infer),
    NMS(),
    ToDetections(),
])
```

#### Outside a pipeline definition: `a >> b`

Use this outside a pipeline definition to join existing named pipelines.

```python
decode = Pipeline([Decode()])
preprocess = Pipeline([Resize((640, 640)), Normalize()])
infer = Pipeline([Infer("model.onnx"), Extract("boxes", "scores", "classes")])
postprocess = Pipeline([NMS(), ToDetections()])

detection = decode >> preprocess >> infer >> postprocess
```

### Merge

Merging two pipelines produces a single flat pipeline. The original pipelines
lose their individual runtime identity inside the composed result.

Two consequences follow directly:

- **Shared context**: operators across the merged boundary share the same
  `Context`, so `Store`/`Recall` keys are visible on both sides.
- **Flattened operator list**: the source pipeline's current operators are
  placed into a new flat list at build time. Extending the source pipeline
  afterwards has no effect on the merged result.

```
Before

 Pipeline A                Pipeline B
┌──────────────────┐      ┌──────────────────┐
│ Op1 → Op2 → Op3  │  +   │ Op4 → Op5 → Op6  │
└──────────────────┘      └──────────────────┘

After

 Pipeline (A + B)
┌────────────────────────────────────┐
│ Op1 → Op2 → Op3 → Op4 → Op5 → Op6  │
└────────────────────────────────────┘
```

Merge is the right choice when the composed result should behave like one
uniform pipeline.

#### Inside a pipeline definition: `inline(p)` / `Inline(p)`

Use these inside `Pipeline([...])` to merge another named pipeline.

```python
preprocess = Pipeline([Resize((640, 640)), Normalize()])

detection = Pipeline([
    Decode(),
    inline(preprocess),
    Infer("model.onnx"),
    Extract("boxes", "scores", "classes"),
    NMS(),
    ToDetections(),
])
```

#### Outside a pipeline definition: `a + b`

Use this outside a pipeline definition to merge existing named pipelines.

```python
infer_stage = Pipeline([Infer("model.onnx"), Extract("boxes", "scores", "classes")])
project_stage = Pipeline([Recall("transform"), ProjectBoxes(), NMS(), ToDetections()])

detection = (
    Pipeline([Resize((640, 640)), Store("transform", source=1), Pick(0), Normalize()])
    + infer_stage
    + project_stage
)
```

#### In place: `pipeline.extend([...])`

Use this to append more operators directly into an existing flat pipeline.

```python
pipeline = Pipeline([Decode(), Resize((640, 640))])
pipeline.extend([Normalize(), Infer("model.onnx")])
pipeline.extend([Extract("boxes", "scores", "classes"), NMS(), ToDetections()])
```

### Summary Table

| Syntax                    | Operation      | Where                    | Result object          |
|---------------------------|----------------|--------------------------|------------------------|
| `embed(p)` / `Embed(p)`   | join           | inside `Pipeline([...])` | outer pipeline         |
| `inline(p)` / `Inline(p)` | merge          | inside `Pipeline([...])` | outer pipeline         |
| `a >> b`                  | join           | outside definition       | new pipeline           |
| `a + b`                   | merge          | outside definition       | new pipeline           |
| `p.extend([...])`         | in-place merge | outside definition       | same pipeline (`self`) |

### Validation After Composition

All composition forms are compatible with `validate()`.

- `inline` and `+` expand to a flat operator list, so validation sees one
  continuous chain
- `embed` and `>>` validate the outer boundary against the inner pipeline's
  contract

Because `>>` and `embed` hold live references, mutating a joined source
pipeline changes the contract of the composed pipeline. Re-run `validate()`
after such changes and before the next execution.

```python
detection = decode_pipeline >> infer_pipeline
detection.validate()
```
