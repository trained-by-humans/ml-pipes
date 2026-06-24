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

Call it after construction, after extending existing pipeline or composing new ones, or after deployment just before 
starting the pipeline.

> [!WARNING]
> Validation is recommended after any pipeline mutation.
> See [COMPOSITION.md](COMPOSITION.md) for composition semantics.

## What validation checks

All validation failures raise `PipelineValidationError`, which subclasses
`ValueError`.

### 1. Region structure checks

A *region* is a bounded section of the pipeline such as
`Batch ... UnBatch` or `Scatter ... Gather` (see [Regions](REGIONS.md)).
Validation checks that region openers and closers are structurally sound.

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

Context scope follows the runtime boundaries of the pipeline: outer and inner
regions do not share stored keys, and embedded pipelines use their own
isolated context. (See [context-system](ARCHITECTURE.md#context-system)).
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

Type validation checks that the type produced at each boundary (operator output) is compatible
with the type expected by the next boundary (next operator input).

Missing annotations are rejected immediately and identify the offending
operator:

```text
StringToFloat is missing a type annotation for __call__ input
IntToString is missing a return type annotation for __call__
```

Example of compatible boundary:

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

This positional unpacking only supports fixed-length tuple boundaries.
Variadic tuple annotations such as `tuple[int, ...]` remain atomic and are
not treated as multi-parameter pipeline boundaries. Non-positional
`__call__` parameters such as `*args`, keyword-only parameters, and
`**kwargs` are not supported because Pipeline chains operators by argument
position. Use a single tuple-typed parameter instead if the operator should
consume a variadic tuple value atomically.

If a multi-parameter operator uses positional defaults, validation emits a
warning. Pipeline ignores those defaults for dispatch and still treats the
operator as a fixed-arity positional boundary to avoid ambiguity with
tuple-valued outputs.

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

> [!TIP]
> Checks run in order. A failure in an earlier pass stops execution, so later
> passes are not reached.

## Validation and boundaries

Type validation works on operator boundaries. Each operator defines its own
boundary by providing information about its input and output. If an operator
boundary is vague, or the pipeline input type is unclear (for example because
the entry operators are generic), that vagueness can propagate forward and
reduce how much validation can prove.

The default mode validates from the boundary information already available. The
other two modes improve the same validation by tightening those boundaries with
additional information.

### 1. Default mode

```python
contract = pipeline.validate()
```

This is the baseline behavior:

- validation starts at the beginning of the pipeline and walks forward,
  checking compatibility using the boundaries it can resolve during the
  forward pass.
- if there is no clear input type for the pipeline, validation assumes the
  pipeline input type is `Any`.

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

- it gives the forward pass a concrete starting boundary, which can tighten
  the resolved operator boundaries and, as a result, the final pipeline
  boundary,
- it is also an asserted expected input contract, so validation will fail if
  the pipeline is incompatible with that declared input.

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

This is an additional improvement over the default mode.

Validation tries to infer a tighter pipeline boundary from downstream
constraints.

This helps when:

- the entry operator is generic or transitive,
- but later operators force the input to be more specific.

Example:

```python
contract = Pipeline([Scatter(), Store("raw"), Decode(), ...]).validate(
    inference=True
)

assert contract.input_type == list[bytes]
```

> [!CAUTION]
> Backward inference only works while the chain remains transparent enough to
> project constraints backward. Vague or opaque operators stop refinement, and
> the boundary remains looser.

### What validation returns

Validation returns a `TypeContract`:

```python
contract = Pipeline([Decode(), Resize((640, 640))]).validate()

assert contract.input_type is bytes
assert contract.output_type == tuple[ImagePayload, ResizeTransform]
```

- `input_type` is the resolved pipeline input boundary after any tightening.
- `output_type` is the resolved output type of the last operator.

## What strict mode is

`strict=True` adds an additional check on top of the normal validation passes.

Its goal is not to tighten the pipeline boundary. Its goal is to reject
unresolved operator-boundary ambiguity.

In strict mode, every operator must resolve to concrete input and output
boundaries, either through annotations or through `resolve_contract(...)`.
Unresolved `Any` means validation cannot fully reason about that operator.

```python
pipeline.validate(strict=True)
```

### What `strict=True` means today

Strict mode checks operator boundaries locally.

It runs the normal validation passes first, then rejects any operator whose
resolved input or output boundary still contains unresolved `Any`, unless that
ambiguity is explicitly justified through `resolve_contract(...)`.

So today, strict mode means:

- no unresolved operator-boundary ambiguity,
- explicit justification for generic operators.

It does not mean global worst-case pipeline reasoning. A strict-mode failure
means the validator could not fully justify one operator boundary, not
necessarily that the pipeline is definitely unsafe at runtime.

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

Internally, validation needs two things:

1. each operator must expose a boundary the validator can reason about,
2. and those boundaries must be resolved against the current upstream type at
   each point in the pipeline.

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

A signature can say what an operator accepts in general, but a contract can
say what this operator accepts and produces at this exact point in this exact
pipeline.

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

Each operator contributes an input boundary and an output boundary. Validation
resolves them from:

- static signatures from `__call__`,
- dynamic contracts from `resolve_contract(...)`,
- and the current upstream type flowing through the pipeline.

That is how operators like `Store`, `Recall`, `Pick`, `Batch`, `UnBatch`,
`Scatter`, and `Gather` can stay generic while still participating in accurate
validation.

If an operator has neither usable annotations nor a usable dynamic contract,
validation fails immediately.
