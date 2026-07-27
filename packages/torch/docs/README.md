# Torch Execution Domain

`ml_pipes.torch` brings Torch-backed execution and on-device tensor postprocess into `ml-pipes`.

For the full package surface, aliases, and operator signatures, see [`INDEX.md`](./INDEX.md).

## Package Profile

| Dimension       | Classification                                                      |
|-----------------|---------------------------------------------------------------------|
| Role / Function | `Inference (Scaffold)`                                              |
| Task Type       | Mostly `Vision`; reusable across other tensor-based inference flows |
| Data Type       | `Tensors`                                                           |

## Scope And Use Cases

This package owns the explicit Torch execution domain inside `ml-pipes`. Many
of its practical use cases overlap with
[`ml_pipes.tensor`](../../tensor/docs/README.md), the default shared
NumPy-side path for tensor-domain work around runtime stages.

Pick this package instead of, or alongside, `ml_pipes.tensor` when a pipeline
needs to:

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

Torch is a separate execution domain. A pipeline crosses from NumPy into Torch
only when you add an explicit conversion step such as `ToTorch()` or
`ToTorchRegistry()`.

To keep mixed Torch and NumPy pipelines composable, the Torch domain mirrors
the Tensor package value models such as `TorchTensorPayload` or `TorchTensorRegistry`.
That mirroring is what lets Torch stages integrate cleanly with
`ml_pipes.tensor` when a pipeline crosses back into the shared NumPy-side
postprocess path. The main addition is `device`, which keeps placement
explicit while the value is in the Torch domain.

The main crossings are:

- `ToTorch` / `ToTorchRegistry` into the Torch domain
- `ToNumpy` / `ToNumpyRegistry` back into the NumPy-side Tensor domain
- `ToDevice` to move Torch-backed values without leaving the Torch domain

At a high level, mixed pipelines look like this:

```text
┌─────────────────────────────────────────────┐
│ NumPy Domain                                │
├─ Decode -> Resize -> Normalize -> ...       │
└───────┬─────────────────────────────────────┘
        |
        | ToTorch / ToTorchRegistry
        ▼
┌─────────────────────────────────────────────┐
│ Torch Domain                                │
├─ TorchInfer -> Torch... -> Torch... -> ...  │
└───────┬─────────────────────────────────────┘
        |
        | ToNumpy / ToNumpyRegistry
        ▼
┌─────────────────────────────────────────────┐
│ NumPy Domain                                │
├─ ToDetections / Visualization / Logging     │
└─────────────────────────────────────────────┘
```

That split keeps Torch execution explicit while still letting the rest of the
pipeline stay in the packages that already own NumPy-side postprocess and typed
task results.

> [!TIP]
> Use `copy=False` when you want the cheapest Torch/NumPy conversion and shared
> storage is acceptable. Use `copy=True` when you want the converted result to
> be isolated from the source.
>
> `ToNumpy` and `ToNumpyRegistry` materialize NumPy arrays on CPU. Once data
> crosses back into NumPy, device placement is no longer part of the
> value model.

## Using Torch In Pipelines

In practice, Torch usually appears in one of three pipeline shapes:
- Torch-native inference stage inside an otherwise NumPy-oriented pipeline
- Heavy postprocess stage in Torch before hand-off to NumPy
- Custom Torch stage inserted into a larger mixed pipeline

### Torch-Native Inference

Use this shape when the main inference stage is Torch-native, or otherwise
already produces Torch-backed outputs, but the surrounding image preparation
and later task logic already fit the NumPy-side packages. Upstream steps
prepare the input in NumPy, `ToTorch()` crosses into the Torch domain,
`TorchInfer(...)` runs the model, and the outputs return to NumPy once
downstream packages need them.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import Decode, Normalize, Resize
from ml_pipes.torch import ToNumpyRegistry, ToTorch, TorchExtract, TorchInfer

pipeline = Pipeline([
    ...,
    Normalize(),
    ToTorch(device="cpu"),
    TorchInfer(model, input_layout="NCHW"),
    TorchExtract("output_0", as_="scores"),
    ToNumpyRegistry(),
    ...
])
```

Use this when the model is Torch-native but the rest of the pipeline is
already clear and standardized in NumPy.

For models that expect a named input such as `pixel_values=...`, use
`input_name="..."`. If the model returns a mapping of named tensors, those
mapping keys become the runtime output names that `TorchExtract(...)` can
select.

### Torch-Side Postprocess

Use this shape when the postprocess itself is heavy enough to justify
switching to Torch, or when large intermediate tensors should stay on-device
during the postprocess.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import ToSegmentations
from ml_pipes.torch import (
    ToNumpyRegistry,
    ToTorchRegistry,
    TorchMasksToBoxes,
    TorchResizeMasks,
    TorchSigmoid,
    TorchSqueeze,
    TorchSoftmax,
)

pipeline = Pipeline([
    ...,
    ToTorchRegistry(device="cuda:0"),
    TorchSqueeze("class_queries_logits", axis=0),
    TorchSqueeze("masks_queries_logits", axis=0),
    TorchResizeMasks(masks="masks_queries_logits"),
    TorchSoftmax("class_queries_logits", as_="class_probs"),
    TorchSigmoid("masks_queries_logits", as_="mask_probs"),
    ...,
    TorchMasksToBoxes(masks="binary_masks", as_="boxes"),
    ToNumpyRegistry(),
    ToSegmentations(scores="final_scores", classes="class_ids", masks="binary_masks"),
])
```

The canonical example is
[`examples/torch/run_mask2former_torch_postprocess.py`](../../../examples/torch/run_mask2former_torch_postprocess.py),
which keeps mask resizing, query scoring, filtering, winner assignment, and
mask-to-box conversion in Torch before converting back to NumPy segmentations.

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
from ml_pipes.torch import ToDevice, ToTorch, TorchInfer

pipeline = Pipeline([
    ToTorch(device="cpu"),
    ToDevice("cuda:0"),
    TorchInfer(model, input_layout="NCHW"),
])
```

`ToDevice` is intentionally coarse-grained:

- for `TorchTensorPayload`, it moves the payload tensor
- for `TorchTensorRegistry`, it moves all tensors currently stored in the
  registry

That makes it useful for stage-level placement, not for fine-grained
per-tensor scheduling inside one registry.

## Timing And Synchronization

Torch GPU work can be asynchronous. A step may queue device work and return
before the device has finished. If a later operator forces synchronization,
the time can appear there instead.

Use `TorchSynchronizeTensors()` when you want to force synchronization at a
chosen point in the pipeline:

```python
from ml_pipes.core import Pipeline
from ml_pipes.torch import TorchArgMax, TorchInfer, TorchSynchronizeTensors

pipeline = Pipeline([
    ...,
    TorchInfer(model),
    TorchSynchronizeTensors(),
    TorchArgMax("logits", as_="classes"),
])
```

`TorchSynchronizeTensors()` is a pass-through operator. It synchronizes the
Torch device work associated with the current Torch-backed value so the next
timed step sees completed work instead of queued work.

> [!WARNING]
> Without `TorchSynchronizeTensors()`, per-operator GPU timings are only
> suggestive. End-to-end wall-clock time is still real, but step attribution
> may drift to the next operator that forces synchronization.

## Further Reading

- [`INDEX.md`](./INDEX.md) for the full surface catalog
- [`docs/README.md`](../../../docs/README.md) for the shared framework docs index
- [`examples/README.md`](../../../examples/README.md) for runnable pipeline entry points
  - [`examples/torch/run_mask2former_torch_postprocess.py`](../../../examples/torch/run_mask2former_torch_postprocess.py)
  - [`examples/torch/run_mask2former_numpy_postprocess.py`](../../../examples/torch/run_mask2former_numpy_postprocess.py)
