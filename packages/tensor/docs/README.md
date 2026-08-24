# Tensor Postprocess Domain

`ml_pipes.tensor` is the shared NumPy-side tensor workbench between runtime
packages and task packages.

For the full package surface, aliases, and operator catalog, see
[`INDEX.md`](./INDEX.md).

## Package Profile

| Dimension       | Classification                                                          |
|-----------------|-------------------------------------------------------------------------|
| Role / Function | `Inference (Scaffold)`                                                  |
| Task Type       | Mostly `Vision`; reusable across other tensor-returning inference flows |
| Data Type       | `Tensors`                                                               |

## Scope And Use Cases

`ml_pipes.tensor` owns the small, explicit, generic NumPy-side tensor
operators and value types that sit around runtime handoff and shared
postprocess. It does not own runtime execution, model-specific decode logic,
task finalization, or image/media preparation.

Use this package when a pipeline needs a reusable NumPy-side tensor segment
between runtime output handoff and task finalization, especially during model
scaffolding or shared postprocess.

## Design Principles

Every operator in this package is designed to uphold the following properties.
They are not style guidelines; they are what makes operators safe to compose
and swap without side effects.

- **Model-agnostic.** No operator knows which model produced the tensors it
  processes. `Softmax`, `ArgMax`, and `Slice` are generic. Model-specific
  adaptations stay in the pipeline list as individual operators, not inside
  shared infrastructure.
- **Precision-agnostic.** Operators preserve the dtype of their input. A
  pipeline that runs in `float32` runs in `float16` without modifying any
  operator. A preprocessing step such as `Normalize()` sets the working
  precision before data enters the Tensor domain.
- **Runtime-agnostic.** Operators use NumPy. They impose no dependency on
  PyTorch, TensorFlow, or any specific hardware. Runtime packages own the
  inference step; the Tensor domain stays plain NumPy and transfers to any
  compute environment.
- **Backend-explicit.** Unlike the points above, operators are not
  backend-agnostic. They do not pretend to be one backend-polymorphic tensor
  layer. Runtime-specific execution stays in the owning runtime packages.
- **Small and explicit.** Each operator should stay small and explicit in the
  pipeline list and should own one clear tensor step.

## Runtime Types

The Tensor package uses explicit runtime values instead of passing bare
`np.ndarray` objects through every step. `TensorPayload` covers one
model-ready tensor together with layout and dtype metadata, while
`TensorRegistry` covers a named working set of tensors for postprocess.

### TensorPayload

`TensorPayload` is the one-tensor value type.

Use it when one array should move through the pipeline as one value and the
next step needs its layout and working dtype together with the tensor:

- `array`: the NumPy tensor
- `layout`: the semantic axis order such as `NHWC` or `NCHW`
- `dtype`: the working dtype recorded for the payload

It is commonly created by preprocessing operators such as `Normalize()`, then
consumed by operators such as `Infer()`, `ToTorch()`, or `Collate()`.

This is the common runtime handoff:

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Infer
from ml_pipes.vision import Normalize

pipeline = Pipeline([
    Normalize(),  # -> TensorPayload
    Infer(model_path),
])
```

### TensorRegistry

`TensorRegistry` is the named multi-tensor working set for postprocess.

Use it when postprocess needs multiple intermediate tensors to stay named and
aligned. Tensor operators then refine those named tensors step by step until a
task package turns them into the final result.

This is the common postprocess shape:

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Extract
from ml_pipes.tensor import ArgMax, GatherScores

pipeline = Pipeline([
    Extract("scores", as_="class_scores"),  # -> TensorRegistry
    ArgMax("class_scores", as_="classes"),
    GatherScores("class_scores", "classes", as_="scores"),
])
```

## Where Tensor Fits

Tensor sits between runtime packages and task packages. It does not own the
runtime step itself.

The nearby package crossings are:

- `Normalize()` from `ml_pipes.vision` creates a `TensorPayload`
- `Extract()` from `ml_pipes.onnx` creates a `TensorRegistry`
- `ToNumpy()` and `ToNumpyRegistry()` from `ml_pipes.torch` return back to the
  Tensor domain
- Vision operators postprocess, render, or log named prediction tensors while
  keeping them in the registry

At a high level, a common flow looks like this:

```text
┌──────────────────────────────────────────────────────────┐
│ Input / Task Domain                                      │
├─ Decode -> Resize -> Normalize -> ...                    │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorPayload
         ▼
┌──────────────────────────────────────────────────────────┐
│ Runtime Domain                                           │
├─ Infer / TorchInfer / ...                                │
└────────┬─────────────────────────────────────────────────┘
         |
         | Extract / ToNumpyRegistry
         ▼
┌──────────────────────────────────────────────────────────┐
│ Tensor Domain                                            │
├─ Squeeze -> Slice -> Softmax -> ArgMax -> FilterTensors  │
└────────┬─────────────────────────────────────────────────┘
         |
         | Vision postprocess / visualization / logging
         ▼
┌──────────────────────────────────────────────────────────┐
│ Vision Domain                                             │
└──────────────────────────────────────────────────────────┘
```

That split keeps runtime integration, shared tensor postprocess, and
task-specific finalization as separate stages.

## Using Tensor In Pipelines

This section shows only the Tensor-owned slice: start from extracted named
tensors, apply small registry operators to refine them, then hand the final
tensors to a task package.

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Extract
from ml_pipes.tensor import ArgMax, GatherScores, Slice, Squeeze, Transpose
from ml_pipes.vision import ConvertBoxFormat

pipeline = Pipeline([
    ...,
    Extract("output0", as_="preds"),
    Squeeze("preds"),
    Transpose("preds"),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="scores"),
    ArgMax("scores", as_="classes"),
    GatherScores("scores", "classes"),
    ConvertBoxFormat(from_="cxcywh"),
])
```

This is the shape used in
[`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py) and
[`examples/run_rfdetr_nano.py`](../../../examples/run_rfdetr_nano.py).

For the full guide on how to build scaffolding around the model, see
[`SCAFFOLDING.md`](../../../docs/SCAFFOLDING.md).

## Further Reading

- [`INDEX.md`](./INDEX.md) for the full surface catalog
- [`docs/README.md`](../../../docs/README.md) for the shared framework docs index
- [`REGIONS.md`](../../../docs/REGIONS.md) for batching and regions
- [`Torch guide`](../../torch/docs/README.md) for Torch/NumPy crossover pipelines
- [`Vision guide`](../../vision/docs/README.md) for task-specific tensor operations and typed results in vision pipelines
- [`examples/README.md`](../../../examples/README.md) for runnable pipeline entry points
  - [`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py)
  - [`examples/run_rfdetr_nano.py`](../../../examples/run_rfdetr_nano.py)
