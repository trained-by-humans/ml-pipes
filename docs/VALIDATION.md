# Pipeline Validation

## Overview

Validation checks a pipeline before runtime. In the current validation flow,
it checks:

1. region structure
2. context scope
3. operator compatibility
4. strict-mode boundary concreteness, if `strict=True`

Validation also returns the resolved pipeline contract so you can see what
input and output types the pipeline currently exposes.

```python
from ml_pipes.core import Pipeline


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToFloat:
    def __call__(self, value: str) -> float:
        return float(value)


pipeline = Pipeline([IntToString(), StringToFloat()])
contract = pipeline.validate()

assert contract.input_type is int
assert contract.output_type is float
```

## When Validation Runs

Validation never runs unless you ask for it.

```python
pipeline = Pipeline([...], auto_validate=True)  # validate during construction
pipeline.extend([...])                          # re-validates if auto_validate=True
pipeline.validate()                             # explicit validation
```

Use `auto_validate=True` when the pipeline is built incrementally, but note:

- `auto_validate` runs the normal validation flow, not strict mode
- if you want strict validation, call `pipeline.validate(strict=True)`
- after composition or other pipeline changes, an explicit `validate()` call is
  still the clearest checkpoint

## Operator Compatibility

Validation resolves each step's contract and checks that the previous step's
output can feed the next step's input.

Compatible boundaries:

```text
IntToString(value: int) -> str
StringToFloat(value: str) -> float
```

Fixed-length tuple outputs are unpacked positionally when the next operator
takes multiple positional inputs:

```text
IntToPair(value: int) -> tuple[int, str]
PairToBool(number: int, text: str) -> bool
```

Compatible boundaries follow normal assignability intuition for the supported
annotation shapes in this page: the previous step's output can be narrower
than the next step's expected input, as long as it is assignable to that input
boundary.

### Call Signatures (`__call__`)

For ordinary callable operators, the contract comes from the annotated
`__call__` signature:

- the positional input parameters define the step's input boundary
- the return annotation defines the step's output boundary
- a fixed-length tuple return can become a routed multi-value boundary when the
  next step expects the same number of positional inputs
- a single tuple-typed parameter keeps the tuple atomic; variadic tuple
  annotations such as `tuple[int, ...]` are not expanded into multiple pipeline
  inputs
- positional defaults do not weaken the boundary; for pipeline dispatch, a
  multi-parameter operator still requires all positional inputs and validation
  warns to make that explicit

If an ordinary operator is missing input or return annotations, validation
fails immediately.

> [!CAUTION]
> Pipeline chains operators by argument position. A callable step must expose
> at least one positional input parameter. `*args`, keyword-only parameters,
> and `**kwargs` are rejected.

### Operator Contracts (`resolve_contract(...)`)

Some operators cannot express their boundary precisely with a static signature
alone. `resolve_contract(...)` exists for those cases.

Use it when the operator boundary depends on:

- the current upstream type
- stored context annotations
- a transitive or passthrough relationship that static annotations cannot
  express cleanly

`resolve_contract(...)` computes the contract for this exact pipeline
position. It returns `(input_types, output_type)`.

```python
from ml_pipes.context import Context, ContextOp
from typing import Any


class AttachStored(ContextOp[Any, Any]):
    def __init__(self, name: str):
        self.name = name

    def apply(self, current: Any, context: Context) -> tuple[Any, Context]:
        return (current, context.load(self.name)), context

    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        stored_type = stored_annotations.get(self.name)
        if stored_type is None:
            raise error_type(f"{self.name!r} is not available in context")
        return (Any,), tuple[current_output, stored_type]
```

Built-in operators such as `Store`, `Recall`, `Pick`, `Batch`, `UnBatch`,
`Scatter`, and `Gather` participate in validation this way.

> [!NOTE]
> Broad static annotations can still validate, but they usually leave the
> published contract looser. `resolve_contract(...)` becomes necessary when
> you need a tighter contract than the static signature can express.

> [!TIP]
> Most of the time generic contracts do not need `resolve_contract(...)`.
> If the contract is straightforward, using `TypeVar`s, a simple
> `T -> T` for type-preserving operators or `T -> M[T]` for simple
> container-mapping operators is enough.

### How Compatibility Is Checked

Validation resolves one contract per step:

- for ordinary operators, it extracts the annotated `__call__` boundary
- for dynamic operators, it asks `resolve_contract(...)` for the boundary at
  this exact pipeline position
- while doing that, it specializes generics and `TypeVar`-based boundaries
  against the current upstream type whenever it can
- if a `TypeVar` cannot be fully resolved, the published contract falls back to
  its bound or constraint, recursively inside containers

Once those boundaries are resolved, validation matches them left to right:

- the previous step's output boundary is checked against the next step's input
  boundary
- fixed-length tuple outputs can be matched against multi-parameter inputs by
  position

### Compatibility Coverage

Validation is a contract checker over a scoped subset of Python's typing
model. It is not a general-purpose replacement for a static type checker.

Compatibility is directional. Validation asks whether the previous step's
output is assignable to the next step's input, not whether the two annotations
are interchangeable.

| Annotation shape | Examples | Current compatibility coverage |
|---|---|---|
| Broad placeholders | `Any`, `object` | Accepted as broad contracts. In `strict=True`, unresolved `Any` is still rejected at the operator-boundary level. |
| Concrete classes | `str`, `bytes`, `ImagePayload` | Supported through normal subclass compatibility. |
| `TypeVar`s | `T`, `TypeVar("T", bound=Base)` | Supported with bounds and constraints. When the upstream type is concrete, validation specializes `TypeVar`-based boundaries where it can; otherwise the published contract falls back to the bound or constraint. |
| Unions | `A \| B` | Supported directionally. Every produced option must be assignable to the downstream expectation. |
| Fixed-length tuples | `tuple[int, str]` | Supported and can route positionally into multi-parameter operators. |
| Variadic tuples | `tuple[int, ...]` | Supported as single values. They are not expanded into multiple pipeline inputs. |
| Common built-in and `collections.abc` generics | `list[T]`, `set[T]`, `frozenset[T]`, `dict[K, V]`, `Iterable[T]`, `Collection[T]`, `Sequence[T]`, `Mapping[K, V]`, `MutableSequence[T]`, `MutableMapping[K, V]`, `MutableSet[T]`, `type[T]` | Supported with the variance rules implemented by the core annotation matcher. |
| Structural `Protocol`s | `Protocol` with annotated fields and methods | Supported when used as downstream expectations or `TypeVar` bounds. The current supported protocol shape is non-parameterized structural protocols, including annotated data members and `Self`-preserving methods.<br><br>Note: method matching strips the receiver first, then requires exact callable shape: order, keyword-visible names, kinds, and defaults. Parameter annotations are checked contravariantly; return annotations covariantly.<br><br>Not supported: class-object boundaries such as `type[Proto]`. |
| Other typing features | `Annotated`, `Literal`, generic `Protocol[T]`, `ParamSpec`, overload-oriented typing constructs | Not part of the current documented compatibility contract. Some cases may work incidentally, but they are not guaranteed. Prefer a simpler boundary annotation or use `resolve_contract(...)` when you need a more explicit contract. |

For more information regarding call signatures and operator contracts, see
[OPERATORS.md](OPERATORS.md).

If a boundary is incompatible, validation raises with the step label:

```text
Pipeline contract mismatch at 1:PairToBool:
  IntToPair provides tuple[int, str] but PairToBool expects (int, float)
```

## Context Scope

Validation tracks stored keys in the same runtime scope. A `Recall("x")` is
valid only if `"x"` was stored earlier in that scope.

Scope rules:

- a key stored inside a region is visible only inside that region
- a key stored outside a region remains visible after that region closes
- embedded pipelines validate against their own isolated context

Examples:

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Batch, Recall, Store, UnBatch

Pipeline([Store("x"), Recall("x")])                             # valid
Pipeline([Recall("x")])                                         # invalid
Pipeline([Recall("x"), Store("x")])                             # invalid
Pipeline([Batch(size=2), Store("x"), UnBatch(), Recall("x")])   # invalid
Pipeline([Store("x"), Batch(size=2), UnBatch(), Recall("x")])   # valid
```

When a recall fails, validation reports the operator label and the keys
available at that point.

A representative error looks like:

```text
Pipeline step 0:Recall references a key that was not stored: 'x'. Keys available at this point: (none)
```

## Region Structure

Validation checks that region openers and closers are structurally valid. It
rejects:

- unmatched closers
- unmatched openers
- interleaved regions
- directly nested regions of the same kind

Examples:

```text
Batch -> Op -> Op -> UnBatch                valid
Batch -> Op -> Op                           Batch has no matching UnBatch
Op -> UnBatch                               UnBatch has no matching opener
Scatter -> Batch -> Gather                  regions interleave
Batch -> Batch -> UnBatch -> UnBatch        directly nested Batch forbidden
```

A representative error looks like:

```text
Pipeline step 0:Batch has no matching UnBatch
```

For region semantics and built-in region pairs, see [REGIONS.md](REGIONS.md).

## Strict Mode

`strict=True` adds another check on top of normal validation. It does not
change operator compatibility, context checks, or region checks. It inspects
operator boundaries themselves and rejects unresolved ambiguity.

```python
from typing import Any
from ml_pipes.core import Pipeline


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value


Pipeline([VagueOp()]).validate()              # accepted
Pipeline([VagueOp()]).validate(strict=True)   # raises
```

In practice, strict mode means:

- unresolved `Any` in an operator input or output is rejected
- unresolved `Any` inside containers such as `list[Any]` or `tuple[int, Any]`
  is also rejected
- the check is orthogonal to default mode, declared input mode, and inference
- `auto_validate=True` remains non-strict

Generic operators can still pass strict mode if they justify their boundary
concretely through `resolve_contract(...)`.

> [!TIP]
> For a side-effect-only passthrough operator, prefer `SideEffectOp`. It
> already threads the upstream type correctly for validation.

A representative error looks like:

```text
Strict mode violation at 0:VagueOp: input type is unresolved (Any).
  Fix: annotate the parameter with a concrete type, or implement resolve_contract to accept and thread the upstream type dynamically.
```

## Returned Contract and Input Modes

Validation always runs the checks above. Once those checks succeed, it returns
a `TypeContract(input_type=..., output_type=...)`. The returned output type
comes from the last resolved operator boundary. The returned input type is
tightened from the entry boundary, any declared `pipeline_input_type`, and
optionally by backward inference. Strict mode is orthogonal; it adds
validation on top of this flow but does not change how compatibility or input
modes work.

### Default Mode

```python
contract = pipeline.validate()
```

With no declared input and no backward inference, validation starts at `Any`
and can only tighten from the entry boundary the forward pass resolves. In
practice, that is usually the first step's resolved input boundary. It does
not back-propagate constraints from later steps.

If the entry boundary stays vague, validation can still succeed, but the
published pipeline input stays loose:

```python
from typing import Any
from ml_pipes.core import Pipeline


class VagueOp:
    def __call__(self, value: Any) -> Any:
        return value


contract = Pipeline([VagueOp()]).validate()
assert contract.input_type is Any
```

### Declared Input

```python
contract = pipeline.validate(pipeline_input_type=...)
```

Declared input mode seeds forward validation with a known pipeline input type.
This is the direct way to tighten the entry boundary when the first operators
are generic.

The declared input is checked, not just copied. If it conflicts with the entry
boundary, validation fails during compatibility checking.

Declared input can also be partial. Validation combines what you declare with
what the operators require:

```python
from typing import Any
from ml_pipes.core import Pipeline


class ParsePair:
    def __call__(self, value: tuple[int, str]) -> bool:
        return True


contract = Pipeline([ParsePair()]).validate(
    pipeline_input_type=tuple[Any, str]
)

assert contract.input_type == tuple[int, str]
assert contract.output_type is bool
```

### Backward Inference

```python
contract = pipeline.validate(inference=True)
```

Backward inference can tighten the returned input type further by propagating
downstream constraints backward through transitive or contract-driven steps.

```python
from typing import Any
from ml_pipes.core import Pipeline


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class ContractPassthrough:
    def __call__(self, value: Any) -> Any:
        return value

    def resolve_contract(self, current_output, stored_annotations, expand, error_type):
        return (Any,), current_output


contract = Pipeline([ContractPassthrough(), IntToString()]).validate(
    inference=True
)

assert contract.input_type is int
assert contract.output_type is str
```

> [!CAUTION]
> Inference does not run by default, and it does not work through every
> operator. Opaque or partially unresolved boundaries stop backward
> propagation, leaving the published input type looser.

## Errors and Warnings

Most validation failures raise `PipelineValidationError`.

Unsupported bare generic annotations can still fail earlier during annotation
normalization.

Validation also emits `PipelineValidationWarning` when a multi-parameter
operator defines positional defaults, because runtime dispatch ignores those
defaults for pipeline chaining.
