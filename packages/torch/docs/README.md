# Torch Execution Domain

`ml_pipes.torch` adds the primitive building blocks for Torch-oriented
pipelines and the boundary operators needed to compose mixed NumPy + Torch
pipelines explicitly.

Today the package focuses on:

- explicit NumPy/Torch boundary crossing and device placement
- Torch inference with one `TorchTensorPayload` flowing into `TorchInfer`
- Torch-side registry postprocess that you intentionally keep on-device

It does not try to wrap the full Torch ecosystem. Training loops, datasets,
model authoring, and richer framework-specific wrappers still stay outside the
package's current scope.

## Runtime Types

The Torch domain introduces Torch-backed runtime values alongside the NumPy
ones in core `ml_pipes`.

The Torch-specific runtime values are:

- `TorchTensorPayload`: one tensor plus layout, dtype, and device metadata
- `TorchTensorRegistry`: mutable named store for intermediate Torch tensors
- `TorchRuntimeOutputs`: named multi-output Torch inference result

Compared to the NumPy-side runtime values, the important addition is `device`.
That metadata is part of the value itself, which means a pipeline can control
where the current Torch-backed data sits and can move whole Torch stages
between CPU, CUDA, or other supported Torch devices explicitly.

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
- for `TorchTensorRegistry`, it moves all tensors currently stored in the registry

That means `ToDevice` effectively moves all upstream Torch tensors that are
still part of the flowing value. It is useful for stage-level device placement,
not for fine-grained per-tensor placement inside one registry.

## Crossing Domains

The important design choice is still that NumPy and Torch remain separate
domains. A pipeline crosses from one to the other only when you add a boundary
operator.

The boundary operators are:

- `ToTorch` / `ToTorchRegistry`
- `ToNumpy` / `ToNumpyRegistry`
- `ToDevice`

At a high level, mixed pipelines would look like this:

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

Example boundary crossing:

```python
from ml_pipes.core import Pipeline
from ml_pipes.tensor import ArgMax, GatherRows, Slice
from ml_pipes.vision import Decode, MapPredictionsToObjects, Normalize, Resize, ToDetections
from ml_pipes.torch import ToNumpyRegistry, ToTorch, TorchExtract, TorchInfer

pipeline = Pipeline([
    Decode(),
    Resize((640, 640)),
    Normalize(),
    ToTorch(device="cuda:0"),
    TorchInfer(model, input_layout="NCHW", output_names=("logits",), output_layouts=("NCHW",)),
    TorchExtract("logits", as_="preds"),
    ToNumpyRegistry(),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="class_scores"),
    ArgMax("class_scores", as_="classes"),
    GatherRows("class_scores", "classes", as_="scores"),
    ToDetections(boxes="boxes", scores="scores", classes="classes"),
    MapPredictionsToObjects(fields={"score": "scores", "class_id": "classes", "box": "boxes"}),
])
```

This keeps the Torch stage focused on inference, then hands the outputs back to
NumPy for the remaining detection logic.

> [!TIP]
> Use `copy=False` when you want the cheapest boundary crossing and shared
> storage is acceptable. Use `copy=True` when you want the converted result to
> be isolated from the source.

> [!NOTE]
> `ToNumpy` and `ToNumpyRegistry` materialize NumPy arrays on CPU. Once data
> crosses into the NumPy boundary, device placement is no longer part of the
> value model.

## Timing Boundaries

Torch GPU work can be asynchronous. A step may queue device work and return
before the device has finished. If a later operator forces synchronization,
the time can appear there instead.

Use `TorchSynchronizeTensors()` when you want to force synchronization at a
chosen boundary:

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

## Common Pipeline Shapes

### Pattern 1: Inference In Torch

This is the simplest handoff. Keep image decode / resize / normalize in NumPy,
run the model in Torch, then hand the outputs back to NumPy.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import Decode, Normalize, Resize
from ml_pipes.torch import ToNumpyRegistry, ToTorch, TorchExtract, TorchInfer

pipeline = Pipeline([
    Decode(),
    Resize((640, 640)),
    Normalize(),
    ToTorch(device="cpu"),
    TorchInfer(model, input_layout="NCHW", output_names=("logits",), output_layouts=("NCHW",)),
    TorchExtract("logits", as_="scores"),
    ToNumpyRegistry(),
])
```

> [!TIP]
> Use this when the model is Torch-native but the rest of the pipeline is
> already clear and standardized in NumPy.

### Pattern 2: Postprocess In Torch

You can keep postprocess in Torch when it gives you a concrete benefit. The two common reasons are:

- Avoiding device transfers for large intermediate tensors
- Leveraging Torch operators that already map well to the target device

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import ToSegmentations
from ml_pipes.torch import ToNumpyRegistry, TorchMasksToBoxes, TorchResizeMasks, TorchSigmoid, TorchSoftmax

pipeline = Pipeline([
    ...,
    TorchResizeMasks(masks="masks_queries_logits"),
    TorchSoftmax("class_queries_logits", as_="class_probs"),
    TorchSigmoid("masks_queries_logits", as_="mask_probs"),
    ...,
    TorchMasksToBoxes(masks="binary_masks", as_="boxes"),
    ToNumpyRegistry(),
    ToSegmentations(scores="final_scores", classes="class_ids", masks="binary_masks"),
])
```

> [!TIP]
> Apply this when you want postprocess stages like NMS, mask processing, top-k
> filtering, or large reductions to stay on-device.
>
> The canonical example is
[`examples/torch/run_mask2former_torch_postprocess.py`](../../../examples/torch/run_mask2former_torch_postprocess.py),
which keeps mask resizing, query scoring, filtering, winner assignment, and
mask-to-box conversion in Torch before converting back to NumPy segmentations.


### Pattern 3: Mid-Stage Torch Handoff

Sometimes only part of the computation benefits from Torch. In that case,
convert the registry, run the Torch-specific part, and hand the remaining
computation back to NumPy.

```python
from ml_pipes.core import Pipeline
from ml_pipes.vision import MasksToBoxes
from ml_pipes.torch import ToNumpyRegistry, TorchResizeMasks, TorchSigmoid, TorchSoftmax

pipeline = Pipeline([
    ...,
    TorchResizeMasks(masks="masks_queries_logits"),
    TorchSoftmax("class_queries_logits", as_="class_probs"),
    TorchSigmoid("masks_queries_logits", as_="mask_probs"),
    ...,
    ToNumpyRegistry(),
    MasksToBoxes(masks="binary_masks", as_="boxes"),
])
```

This keeps the rest of the pipeline unchanged while inserting a Torch stage
midway through an otherwise NumPy-oriented flow.

### Choosing Where A Stage Runs

Run a stage in Torch when Torch gives that stage a real advantage:

- the needed convenience operators already exist under `ml_pipes.torch`
- keeping the stage on-device meaningfully improves throughput or latency
- the stage reads more clearly as a Torch tensor pipeline

Run a stage in NumPy when NumPy is the better fit:

- the remaining steps are already implemented and tested in NumPy
- you need projection, visualization, or other NumPy-oriented utilities
- staying in Torch would add conversion or device complexity without a payoff

> [!WARNING]
> Frequent domain crossing can erase the benefit of using Torch in the first
> place. If a stage only gains a small benefit from Torch, repeated
> `ToTorch...` / `ToNumpy...` boundaries can cost more than they save.

The paired Mask2Former examples are the best concrete reference:

- [`examples/torch/run_mask2former_torch_postprocess.py`](../../../examples/torch/run_mask2former_torch_postprocess.py)
- [`examples/torch/run_mask2former_numpy_postprocess.py`](../../../examples/torch/run_mask2former_numpy_postprocess.py)

## Quick Reference

Torch operator groups:

- conversion / boundary:
  `ToTorch`, `ToTorchRegistry`, `ToNumpy`, `ToNumpyRegistry`, `ToDevice`
- execution / runtime:
  `TorchInfer`, `TorchExtract`, `TorchCollate`, `TorchDistribute`, `TorchAsType`
- synchronization:
  `TorchSynchronizeTensors`
- tensor math / indexing / filtering:
  `TorchArgMax`, `TorchGatherRows` (public alias: `TorchGatherScores`), `TorchTopK`,
  `TorchTopKIndices2D`, `TorchSlice`, `TorchSoftmax`, `TorchSigmoid`,
  `TorchMultiplyTensors`, `TorchCreateTensorMask`,
  `TorchCreateTensorMaskByThreshold`,
  `TorchApplyTensorMask`, `TorchSelectTensors`, `TorchFilterTensorsByScore`,
  `TorchFilterTensorsByClasses`, `TorchFilterTensorsByMasksArea`,
  `TorchSortTensorsBy`
- segmentation helpers:
  `TorchWeightMasksByScores`, `TorchResizeMasks`, `TorchMeanMaskScores`,
  `TorchMasksToBoxes`, `TorchNMS`

Implementation and exports live in:

- [`packages/torch/src/ml_pipes/torch/__init__.py`](../src/ml_pipes/torch/__init__.py)
- [`packages/torch/src/ml_pipes/torch/ops.py`](../src/ml_pipes/torch/ops.py)

Behavioral examples live in:

- [`tests/test_torch.py`](../../../tests/test_torch.py)

## Why Torch Is A Separate Domain

In ml-pipes Torch is a separate, explicit execution domain. A common alternative is a polymorphic tensor layer where the same pipeline
value can silently be backed by NumPy, Torch, TensorRT-style buffers, or some
other runtime tensor object. This repo does not take that route.

Instead, NumPy and Torch stay as explicit domains with explicit crossings.

That choice keeps the system simpler:

- the active runtime is visible in the pipeline list
- optional dependencies stay optional
- library behavior does not depend on hidden backend dispatch
- NumPy-oriented utilities do not need to pretend to be generic tensor kernels
- Torch-oriented stages can still be added where they help

The goal is not to hide backend differences behind one polymorphic tensor type.
The goal is to make those differences composable and explicit.
