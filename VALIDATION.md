# Pipeline Validation

Validation catches structural, scoping, and type errors in a pipeline before
it runs. A `PipelineValidationError` at construction or deploy time is much
cheaper than a `TypeError` or `KeyError` buried inside a batch at runtime.

## How to run validation

Validation never runs automatically unless you ask for it.

### At construction

```python
pipeline = Pipeline([...], auto_validate=True)
```

The full pipeline is validated during `__init__`. Any error raises before the
pipeline is returned.

### After `extend()`

```python
pipeline = Pipeline([Store("x")], auto_validate=True)
pipeline.extend([Recall("x")])   # re-validates automatically
```

When `auto_validate=True`, every `extend()` call re-validates the full
pipeline.

### Explicit call

```python
pipeline.validate()
```

Call it after construction, after extending, or before deployment.

This is especially important for joined pipelines. `>>` and `embed()` hold
live references to the original pipeline objects, so mutating one side after
composition can change the effective contract without the outer pipeline
knowing.

```python
detector = Pipeline([Infer("yolo.onnx"), Extract("output0")])
pipeline = preprocess >> detector

detector.extend([ToDict()])   # custom operator: output type changes

pipeline.validate()
```

`+` and `inline()` copy operators at construction time, so they do not carry
those live-reference contract changes forward.

> [!WARNING]
> Recommended before the first production run and after any mutation of a
> joined pipeline. See [COMPOSITION.md](COMPOSITION.md).

## What validation checks

Checks run in order. A failure in an earlier pass stops execution, so later
passes are not reached.

All validation failures raise `PipelineValidationError`, which subclasses
`ValueError`.

### 1. Region structure checks

Validation first checks that region openers and closers are structurally
sound.

It rejects:

- unmatched closers,
- unmatched openers,
- interleaved regions,
- directly nested regions of the same kind.

Examples:

```
Batch -> Op -> Op -> UnBatch                valid
Batch -> Op -> Op                           Batch has no matching UnBatch
Op -> UnBatch                               UnBatch has no matching opener
Scatter -> Batch -> Gather                  regions interleave
Batch -> Batch -> UnBatch -> UnBatch        directly nested Batch forbidden
```

This pass is purely structural. It does not look at types yet.

### 2. Context scoping checks

Validation then walks the operator list and tracks which keys are available at
each point.

A `Recall("x")` is only valid if `"x"` was previously stored in the same
scope.

Scope rules:

- A key stored inside a region is only visible inside that region.
- A key stored outside a region remains visible after that region closes.
- `Embed` validates the inner pipeline separately, so inner and outer contexts
  are isolated.

Examples:

```python
Pipeline([Store("x"), Recall("x")])                         # valid
Pipeline([Recall("x")])                                     # invalid
Pipeline([Recall("x"), Store("x")])                         # invalid
Pipeline([Batch(size=2), Store("x"), UnBatch(), Recall("x")])  # invalid
Pipeline([Store("x"), Batch(size=2), UnBatch(), Recall("x")])  # valid
```

When a recall fails, validation reports the operator label and the keys that
were available at that point:

```text
Recall('transform') at 3:Recall references a key that was not stored.
Keys available at this point: ['features', 'metadata']
```

If no keys were available:

```text
Keys available at this point: (none)
```

If the failure happens inside `Embed`, the outer error wraps the inner one so
the attribution stays clear.

### 3. Type validation checks

Type validation checks that the type produced at each boundary is compatible
with the type expected by the next boundary.

Missing annotations are rejected immediately and identify the offending
operator:

```text
StringToFloat is missing a type annotation for __call__ input
IntToString is missing a return type annotation for __call__
```

Examples:

```text
IntToString(value: int) -> str
str -> str
StringToFloat(value: str) -> float
```

Tuple outputs are unpacked automatically when the next operator takes multiple
positional parameters:

```text
IntToPair(value: int) -> tuple[int, str]
tuple[int, str] -> (int, str)
PairToBool(number: int, text: str) -> bool
```

Broader downstream input types are allowed:

```text
str -> object
```

So an operator producing `str` can feed one that accepts `object`.

If a boundary is incompatible, validation raises with the operator label:

```text
Pipeline contract mismatch at 1:PairToBool:
  IntToPair returns (int, str) but PairToBool expects (int, float)
```

## How boundary tightening is configured

Boundary tightening is a validation configuration concern.

Type validation is only as accurate as the boundary it can reason from. If the
pipeline starts from `Any`, that vagueness can propagate forward and reduce how
much validation can prove.

So validation tries to tighten the pipeline boundary before and during type
checking. The returned `TypeContract.input_type` is the result of that
tightening, not the reason for it.

Validation exposes three modes.

### 1. Default forward tightening mode

```python
contract = pipeline.validate()
```

This is the baseline behavior:

- validation starts from `Any`,
- and tightens the pipeline boundary only if the first concrete entry boundary
  it can determine is more specific than `Any`.

Example:

```python
contract = Pipeline([Decode(), Resize((640, 640))]).validate()
assert contract.input_type is bytes
```

If the first operator boundary is vague, validation may still succeed, but the
pipeline boundary stays loose:

```python
contract = Pipeline([Pick(0)]).validate()
assert contract.input_type is Any
```

### 2. Declared pipeline input mode

```python
contract = pipeline.validate(pipeline_input_type=ImagePayload)
```

This is the direct way to improve validation accuracy when the entry boundary
is otherwise vague.

The declared type does two things:

- it tightens the pipeline boundary up front,
- and it is checked as the declared pipeline input for the pipeline.

That means an incorrect declaration can fail validation, which is exactly what
you want.

Example:

```python
contract = Pipeline([Normalize(), Infer("model.onnx")]).validate(
    pipeline_input_type=ImagePayload
)
assert contract.input_type is ImagePayload
```

### 3. Backward inference mode

```python
contract = pipeline.validate(inference=True)
```

This is an additional improvement over the default forward pass.

Validation tries to infer a tighter pipeline boundary from downstream
constraints.

This helps when:

- the entry operator is generic or transitive,
- but later operators force the input to be more specific.

Example:

```python
contract = Pipeline([Scatter(), Decode(), Gather()]).validate(
    inference=True
)

assert contract.input_type == list[bytes]
assert contract.output_type == list[ImagePayload]
```

Backward inference only works while the chain remains transparent enough to
project constraints backward. Vague or opaque operators stop refinement, and
the boundary remains looser.

### What validation returns

Validation returns a `TypeContract`:

```python
contract = Pipeline([Decode(), Resize((640, 640))]).validate()

assert contract.input_type is bytes
assert contract.output_type == tuple[ImagePayload, ResizeTransform]
```

- `input_type` is the result of boundary tightening.
- `output_type` is the resolved output type of the last operator.

## What strict mode checks

`strict=True` adds an additional check on top of the normal validation passes.

Its goal is not to tighten the pipeline boundary. Its goal is to reject vague
operator boundaries.

In strict mode, every operator must resolve to concrete input and output
boundaries, either through annotations or through `resolve_contract(...)`.
Unresolved `Any` means validation cannot fully reason about that operator.

```python
pipeline.validate(strict=True)
```

Strict mode is orthogonal to boundary tightening:

- it operates on resolved operator boundaries,
- it does not care whether the final pipeline `input_type` came from default
  tightening, an explicit `pipeline_input_type`, or backward inference.

Generic operators can still satisfy strict mode if they resolve concretely from
the upstream type:

```python
class LogDetections:
    def __call__(self, payload: Any) -> Any:
        ...

    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        return (Any,), current_output
```

This is enough for strict mode as long as `resolve_contract(...)` produces
concrete boundaries at validation time.

> [!TIP]
> For side-effect operators that accept any input and return it unchanged,
> subclass `SideEffectOp` instead of writing the passthrough logic yourself.

Strict-mode failures are explicit:

```text
Strict mode violation at 2:LogOp: input type is unresolved (Any).
  Fix: annotate the parameter with a concrete type, or implement resolve_contract
  to accept and thread the upstream type dynamically.
```

```text
Strict mode violation at 2:LogOp: output type is unresolved (Any).
  Fix: annotate the return type with a concrete type, or implement resolve_contract
  to return the upstream type (e.g. passthrough: return (Any,), current_output).
```

## How validation works internally

Validation is built from two ideas:

1. Each operator must expose a boundary the validator can reason about.
2. The validator threads those boundaries through the pipeline and tries to
   tighten the pipeline boundary so the reasoning stays accurate.

### The two foundations of type validation

#### 1. Operator signature boundaries

The first foundation is the operator signature: the annotated input parameters
and return type on `__call__`.

For a normal typed operator, the signature is enough:

```python
class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)
```

From that alone, validation can resolve:

- input boundary: `int`
- output boundary: `str`

This is the simplest and preferred case.

#### 2. Operator contract boundaries

Some operators cannot be described accurately by a static signature alone.

Examples:

- operators that accept many shapes of input,
- operators whose output depends on the upstream type,
- operators that depend on stored context,
- operators whose boundary cannot be described precisely by a single static signature.

Those operators need a dynamic contract through `resolve_contract(...)`.

```python
def resolve_contract(
    self,
    current_output,
    stored_annotations,
    expand_output_annotation,
    validation_error_type,
):
    ...
```

Why this is needed:

- a signature can say what an operator accepts in general,
- but a contract can say what this operator accepts and produces at this exact
  point in this exact pipeline.

Example:

```python
class PassthroughOp:
    def __call__(self, value: Any) -> Any:
        return value

    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        return (Any,), current_output
```

The static signature is vague. The dynamic contract is precise: "I accept
anything, and I return exactly what came in."

That is why contracts exist. Without them, generic and context-sensitive
operators would poison validation with `Any`.

### How boundaries are resolved and matched

Each operator contributes an input boundary and an output boundary.

Validation resolves them from:

- static signatures from `__call__`,
- dynamic contracts from `resolve_contract(...)`,
- and the current upstream type flowing through the pipeline.

That is how operators like `Store`, `Recall`, `Pick`, `Batch`, `UnBatch`,
`Scatter`, and `Gather` can stay generic while still participating in accurate
validation.

If an operator has neither usable annotations nor a usable dynamic contract,
validation fails immediately.

In practice:

- signatures are the default source of truth,
- contracts refine or replace signatures when the operator is dynamic,
- and the upstream type is what lets contracts specialize to the current
  pipeline position.
