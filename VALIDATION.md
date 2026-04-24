# Pipeline Validation

Validation catches structural and type errors in a pipeline before it runs.
The earlier an error is found, the cheaper it is — a `PipelineValidationError`
at construction or deploy time is far better than a `TypeError` or `KeyError`
buried inside a batch at runtime.

Validation answers three questions about a pipeline:

1. Are the `Batch`/`UnBatch` regions correctly paired and non-nested?
2. Does every `Recall` reference a key that was previously stored in the same scope?
3. Does the output type of each operator match the input type expected by the next?

## Triggers

Validation never runs automatically unless you ask for it. There are three ways
to trigger it:

**Explicit call**

```python
pipeline.validate()
```

Call this any time — after construction, after extending, or before a
deployment. Recommended before the first production run and after any mutation
of a joined pipeline (see [COMPOSITION.md](COMPOSITION.md)).

**At construction**

```python
pipeline = Pipeline([...], auto_validate=True)
```

Validates immediately after `__init__`. Any error raises before the pipeline
object is returned to the caller.

**After `extend()`**

```python
pipeline = Pipeline([Store("x")], auto_validate=True)
pipeline.extend([Recall("x")])   # re-validates automatically
```

When `auto_validate=True`, every `extend()` call re-validates the full
pipeline. When `auto_validate=False` (the default), `extend()` is silent.

## Validations

The three checks always run in order. A failure in an earlier check stops
execution — later checks are not reached.

### 1. Batch pairing

Walks the operator list and checks that every `Batch` has a matching `UnBatch`
and vice versa. Nesting is not allowed — a `Batch` inside a `Batch` region
raises immediately.

```
Batch → Op → Op → UnBatch      ✅ valid
Batch → Op → Op                ❌ Batch has no matching UnBatch
Op → UnBatch                   ❌ UnBatch at position N has no matching Batch
Batch → Batch → UnBatch        ❌ Nested Batch regions are not supported
```

### 2. Context interactions

Walks the operator list and tracks which keys have been stored at each point.
A `Recall` is only valid if its key was stored earlier in the same scope.

Scope rules:

- `Store` inside a `Batch` region is only visible inside that region — it is
  discarded when `UnBatch` is reached.
- Keys stored outside a `Batch` region survive across it and are visible after
  `UnBatch`.
- `Embed` runs with a fully isolated context — outer keys are invisible inside
  an embedded pipeline, and inner keys do not leak out.

```python
Pipeline([Store("x"), Recall("x")])                         # ✅ correct order
Pipeline([Recall("x")])                                     # ❌ key not stored
Pipeline([Recall("x"), Store("x")])                         # ❌ Recall before Store
Pipeline([Batch(...), Store("x"), UnBatch(), Recall("x")])  # ❌ key out of scope
Pipeline([Store("x"), Batch(...), UnBatch(), Recall("x")])  # ✅ outer key survives
```

### 3. Type contract

Walks the operator list and threads the output type of each operator into the
input type expected by the next. A mismatch raises with a message identifying
the exact boundary.

- **Annotations required** — every operator must annotate its `__call__`
  parameters and return type. Missing annotations raise before any type
  comparison is attempted.

  ```
  IntToString(value: int) → str
  │
  str    ✅  output == input
  │
  StringToFloat(value: str) → float
  ```

- **Tuple unpacking** — a `tuple[int, str]` output is automatically unpacked
  into the next operator's `(number: int, text: str)` parameters.

  ```
  IntToPair(value: int) → tuple[int, str]
  │
  tuple[int, str]   ✅  unpacked into → (int, str) 
  │
  PairToString(number: int, text: str) → str
  ```

- **Store / Recall type threading** 
  `Store` records the annotation of the saved value.
  ```
  str, float
  │
  Store("label", index=0)    ✅ Store str as "label"
  │
  str, float
  ```
  `Recall` injects it back into the type stream so downstream
  operators see the correct types.
  ```
  int                # later down the pipeline
  │
  Recall("label")              
  │
  tuple[int, str]    ✅  inject stored parameter properly
  ```

- **Embed boundary check** — the inner pipeline is validated independently
  and then its first input type is checked against the outer pipeline's
  current output type at the join point.

  ```
  IntToString(value: int) → str
  │
  str → str    ✅  output == embed first input
  │
  embed(StringToFloat(value: str) → float)

  IntToString(value: int) → str
  │
  str → float    ❌  output ≠ embed first input
  │
  embed(FloatToBool(value: float) → bool)
  ```

- **Covariant assignment** — broader downstream input types are accepted; an
  operator returning `str` feeding into one that expects `object` is valid.

  ```
  IntToString(value: int) → str
  │
  str → object      ✅  str ⊆ object
  │
  ObjectConsumer(value: object) → object
  ```

## Error messages

All errors are raised as `PipelineValidationError(ValueError)`.

**Operator index and class name** appear in every message so the failing
boundary can be located immediately without counting operators manually:

```
Pipeline contract mismatch at 2:StringToFloat:
  IntToString returns str but StringToFloat expects float
```

**Available keys** are listed when a `Recall` references a missing key:

```
Recall('transform') at 3:Recall references a key that was not stored.
Keys available at this point: ['features', 'metadata']
```

When no keys have been stored yet:

```
Keys available at this point: (none)
```

**Embed attribution** wraps the inner error so the outer position is clear:

```
Validation error inside 1:Embed: Recall('x') at 0:Recall references a key
that was not stored. Keys available at this point: (none)
```

**Missing annotations** name the offending operator:

```
StringToFloat is missing a type annotation for __call__ input
IntToString is missing a return type annotation for __call__
```

## Strict mode

`strict=True` adds a fourth check on top of the normal three. It rejects any
operator whose resolved input or output type is still `Any` after
`_resolve_type_contract` has run with the real upstream types.

```python
pipeline = Pipeline([...], strict=True)
pipeline.validate()
```

The check runs on resolved types, not raw annotations. This means an operator
can declare `Any` in its annotations and still pass strict mode, as long as it
implements `resolve_contract` to return a concrete type based on what it
actually receives:

```python
class LogDetections(ContextOp):
    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        return (Any,), current_output  # accept anything, return what I received
```

`Store`, `Recall`, `Pick`, `Batch`, and `UnBatch` are exempt from the input
check — their `(Any,)` input contract is architectural. Their output types are
still checked.

**Error messages**

Vague input — checked first:
```
Strict mode violation at 2:LogOp: input type is unresolved (Any).
  Fix: annotate the parameter with a concrete type, or implement resolve_contract
  to accept and thread the upstream type dynamically.
```

Vague output — only reached when input is concrete:
```
Strict mode violation at 2:LogOp: output type is unresolved (Any).
  Fix: annotate the return type with a concrete type, or implement resolve_contract
  to return the upstream type (e.g. passthrough: return (Any,), current_output).
```

## Composition and live references

`>>` and `embed()` hold live references to the original pipeline objects.
Mutating a joined pipeline after composition changes its type contract without
the outer pipeline knowing. Always call `validate()` after any such mutation
and before the next execution.

```python
detector = Pipeline([Infer("yolo.onnx"), Extract("output0")])
pipeline = preprocess >> detector

detector.extend([SerializeToDict()])   # output type changes to dict

pipeline.validate()   # catches the broken boundary before it reaches production
```

`+` and `inline()` copy operators at construction time — there are no live
references and no silent contract changes after the fact.

See [COMPOSITION.md](COMPOSITION.md) for the full composition semantics.
