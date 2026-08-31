# Torch Execution Domain

`ml_pipes.torch` brings Torch-backed execution and on-device tensor postprocess into `ml-pipes`.

For the full package surface, aliases, and operator signatures, see [`INDEX.md`](./INDEX.md).

## Package Profile

| Dimension       | Classification                                                      |
|-----------------|---------------------------------------------------------------------|
| Role / Function | `Inference (Scaffold)`                                              |
| Task Type       | General; reusable across tensor-based inference flows                |
| Data Type       | `Tensors`                                                           |

## Scope And Use Cases

This package owns the explicit Torch execution domain inside `ml-pipes`. Many
of its practical use cases overlap with the NumPy-side package chain around
runtime and postprocess.

Its scope covers two Torch-native slices:

- a model runtime boundary parallel to
  [`ml_pipes.onnx`](../../onnx/docs/README.md)
- generic tensor postprocess that mirrors
  [`ml_pipes.tensor`](../../tensor/docs/README.md)

Pick this package instead of, or alongside, that NumPy-side package chain when
a pipeline needs to:

- run a `torch.nn.Module` as one explicit inference stage
- keep control over device placement for a heavy postprocess stage on GPU
- keep a large tensor stage in Torch to avoid unnecessary device transfers or
  host-device synchronization
- reuse existing PyTorch operators or postprocess code inside the pipeline

For broader background on the Torch vs NumPy runtime tradeoff, see the
[APXML PyTorch overview](https://apxml.com/courses/getting-started-with-pytorch/chapter-1-pytorch-fundamentals-setup/what-is-pytorch).

> [!NOTE]
> The Torch package does not try to cover training loops, datasets, model
> authoring, or framework-specific convenience layers beyond explicit Torch
> execution and postprocess.

## Design Principles

- Keep Torch and NumPy as separate explicit domains.
- Make device placement part of the runtime value instead of hidden state.
- Make Torch/NumPy conversions visible in the pipeline list.
- Keep the package thin and pipeline-centered rather than wrapping all of
  PyTorch.

## Where Torch Fits

Torch is an explicit execution domain for the parts of a pipeline that benefit
from a Torch model or Torch-native tensor computation. A pipeline can enter for
inference, keep a generic postprocess segment on-device, and return to NumPy
once later stages fit `ml_pipes.tensor` or `ml_pipes.vision` better.

- **Torch values** — `torch.TensorPayload` and `torch.TensorRegistry` mirror
  Tensor values, while `torch.RuntimeOutputs` mirrors the ONNX runtime handoff;
  each records its `device`.
- **Conversion** — enter the Torch domain with `ToTorch()` or
  `ToTorchRegistry()`, and return to NumPy with `ToNumpy()` or
  `ToNumpyRegistry()`.
- **Available operators** — once a value is in the appropriate Torch type, use
  `Infer` for a Torch module, `Extract` / `Distribute` for runtime outputs, and
  the generic Tensor-mirror operators for `torch.TensorRegistry`.

For example, a Torch-native inference segment with optional Torch-side
postprocess would look like this:

```text
┌──────────────────────────────────────────────┐
│ NumPy Domain                                 │
├─ Decode -> Resize -> Normalize -> ...        │
└───────┬──────────────────────────────────────┘
        |
        | ToTorch
        ▼
┌──────────────────────────────────────────────┐
│ Torch Domain                                 │
├─ Infer -> generic Tensor operators -> ...    │
└───────┬──────────────────────────────────────┘
        |
        | ToNumpyRegistry
        ▼
┌──────────────────────────────────────────────┐
│ NumPy Domain                                 │
├─ Visualization / Logging / Task finalization │
└──────────────────────────────────────────────┘
```

That split keeps Torch execution explicit while still handing off to the
packages that own visualization, logging, and task finalization.

## Using Torch In Pipelines

### Torch-Native Inference

Use this shape when preparation is NumPy-side but inference runs through a
`torch.nn.Module`. `ToTorch()` crosses one input payload into the Torch
domain, `Infer(...)` runs the model, and `Extract(...)` exposes the outputs as
a `torch.TensorRegistry`. Continue with optional generic Torch postprocessing
there, then use `ToNumpyRegistry()` before downstream packages need NumPy
values.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import Decode, Normalize, Resize
from ml_pipes.torch import ToNumpyRegistry, ToTorch, Extract, Infer

pipeline = Pipeline([
    ...,
    Normalize(),
    ToTorch(device="cpu"),
    Infer(model, input_layout="NCHW"),
    Extract("output_0", as_="scores"),
    # Optional generic postprocessing in Torch (Softmax, FilterTensors, etc.)
    ...,
    ToNumpyRegistry(),
    # NumPy-side task finalization (ProjectBoxes, DrawMasks, etc.)
    ...,
])
```

For models that expect a named input such as `pixel_values=...`, use
`input_name="..."`. If the model returns a mapping of named tensors, those
mapping keys become the runtime output names that `Extract(...)` can
select.

### Custom Torch Stage In A Mixed Pipeline

Use this shape when one part of the computation already exists as local
Torch code, or depends on a custom Torch operator that you want to reuse.
Convert into the Torch domain for that stage, run the custom Torch logic,
then convert back to NumPy so the rest of the pipeline can stay unchanged.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import MasksToBoxes
from ml_pipes.torch import ToNumpyRegistry, ToTorchRegistry

pipeline = Pipeline([
    ...,
    ToTorchRegistry(device="cuda:0"),
    CustomTorchStage(),  # local Torch operator or wrapped custom code
    ToNumpyRegistry(),
    ...
])
```

This keeps the rest of the pipeline unchanged while inserting one custom Torch
stage midway through an otherwise NumPy-oriented flow.

> [!WARNING]
> Frequent Torch/NumPy conversions can erase the benefit of using Torch in the
> first place. If a stage only gains a small benefit from Torch, repeated
> `ToTorch...` / `ToNumpy...` conversions can cost more than they save.

## Device Placement

Use `ToDevice` when you want to move Torch payloads or Torch registries without
leaving the Torch domain:

```python
from ml_pipes.core import Pipeline
from ml_pipes.torch import ToDevice, ToTorch, Infer

pipeline = Pipeline([
    ToTorch(device="cpu"),
    ToDevice("cuda:0"),
    Infer(model, input_layout="NCHW"),
])
```

`ToDevice` is intentionally coarse-grained:

- for `TensorPayload`, it moves the payload tensor
- for `TensorRegistry`, it moves all tensors currently stored in the
  registry

That makes it useful for stage-level placement, not for fine-grained
per-tensor scheduling inside one registry.

## Conversion And Copy Semantics

Boundary conversion operators such as `ToTorch`, `ToTorchRegistry`,
`ToNumpy`, and `ToNumpyRegistry` all take a `copy=...` flag that controls
whether a conversion may reuse source storage when aliasing is possible, or
whether the converted value should be detached from the source.

For example:

```python
from ml_pipes.core import Pipeline
from ml_pipes.torch import ToNumpyRegistry, ToTorchRegistry

pipeline = Pipeline([
    ...,
    ToTorchRegistry(device="cuda:0", copy=False),
    CustomTorchStage(),
    ToNumpyRegistry(copy=True),
    ...
])
```

Use the flag differently depending on the priority:

- `copy=False`: prefer the cheapest conversion; use this when the crossing should stay as cheap as possible
- `copy=True`: isolate the converted value; use this when the result should be independent from the source

> [!NOTE]
> `ToNumpy` and `ToNumpyRegistry` always materialize NumPy arrays on CPU. Once
> data crosses back into NumPy, device placement is no longer part of the
> value model.

## Timing And Synchronization

Torch GPU work can be asynchronous. A step may queue device work and return
before the device has finished. If a later operator forces synchronization,
the time can appear there instead.

Use `SynchronizeTensors()` when you want to force synchronization at a
chosen point in the pipeline:

```python
from ml_pipes.core import Pipeline
from ml_pipes.torch import ArgMax, Infer, SynchronizeTensors

pipeline = Pipeline([
    ...,
    Infer(model),
    SynchronizeTensors(),
    ArgMax("logits", as_="classes"),
])
```

`SynchronizeTensors()` is a pass-through operator. It synchronizes the
Torch device work associated with the current Torch-backed value so the next
timed step sees completed work instead of queued work.

> [!WARNING]
> Without `SynchronizeTensors()`, per-operator GPU timings are only
> suggestive. End-to-end wall-clock time is still real, but step attribution
> may drift to the next operator that forces synchronization.

## Further Reading

- [`INDEX.md`](./INDEX.md) for the full surface catalog
- [`docs/README.md`](../../../docs/README.md) for the shared framework docs index
- [`examples/README.md`](../../../examples/README.md) for runnable pipeline entry points
  - [`examples/torch/run_mask2former_torch_postprocess.py`](../../../examples/torch/run_mask2former_torch_postprocess.py)
  - [`examples/torch/run_mask2former_numpy_postprocess.py`](../../../examples/torch/run_mask2former_numpy_postprocess.py)
