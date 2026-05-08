# Torch Execution Domain

`ml_pipes.torch` is the explicit Torch execution domain for pipelines that want
Torch-native inference, Torch-native postprocess, or both. Core `ml_pipes`
stays NumPy-oriented. Crossing between the two domains is always explicit.

## Install

```bash
pip install -e .[torch]
```

## Mental Model

There are two first-class tensor domains in the library:

- NumPy under `ml_pipes`
- Torch under `ml_pipes.torch`

Use NumPy when your pipeline is primarily preprocessing, ONNX inference,
projection, visualization, or general Python postprocess. Use Torch when you
want model execution or postprocess to stay in Torch tensors and, when
possible, stay on-device.

Torch-specific runtime values are:

- `TorchTensorPayload`: one tensor plus layout, dtype, and device metadata
- `TorchTensorRegistry`: mutable named store for intermediate Torch tensors
- `TorchRuntimeOutputs`: named multi-output Torch inference result

## Domain Boundaries

The main boundary operators are:

- `ToTorch` / `ToTorchRegistry`
- `ToNumpy` / `ToNumpyRegistry`
- `ToDevice`

Typical boundary crossings look like this:

```python
from ml_pipes import Normalize, Pipeline
from ml_pipes.torch import ToNumpyRegistry, ToTorch, TorchExtract, TorchInfer

pipeline = Pipeline([
    Normalize(),
    ToTorch(device="cuda:0"),
    TorchInfer(model, input_layout="NCHW", output_names=("logits",), output_layouts=("NCHW",)),
    TorchExtract("logits", as_="scores"),
    ToNumpyRegistry(),
])
```

> [!WARNING]
> Mixing NumPy and Torch operators requires an explicit conversion step.

### Conversion And Copy Semantics

`copy` exists on the conversion operators:

- `ToTorch(copy=False)`
- `ToTorchRegistry(copy=False)`
- `ToNumpy(copy=False)`
- `ToNumpyRegistry(copy=False)`

The default is `copy=False`:

- share storage when safe and possible
- avoid redundant copies when conversion already detached the result from the source

With `copy=True`:

- the converted result is isolated from the source
- redundant extra copies are still skipped when device transfer or dtype conversion
  already detached the result from the source

`ToNumpy` and `ToNumpyRegistry` always materialize NumPy arrays on CPU.

> [!WARNING]
> `ToNumpy` converts to CPU. It does not move or mutate the original Torch tensor.

> [!NOTE]
> `copy=True` means the converted result is isolated from the source. It does not
> mean "preserve device" or "clone the original Torch tensor in place".

### Device Movement

Use `ToDevice` when you want to move Torch payloads or Torch registries without
leaving the Torch domain:

```python
from ml_pipes import Pipeline
from ml_pipes.torch import ToDevice, ToTorch, TorchInfer

pipeline = Pipeline([
    ToTorch(device="cpu"),
    ToDevice("cuda:0"),
    TorchInfer(model, input_layout="NCHW"),
])
```

> [!WARNING]
> Torch GPU work can be asynchronous. For profiling or step-level latency
> attribution, insert `TorchSynchronizeTensors()` at the boundary you want to
> measure. Without it, time may appear on a later operator that forces
> synchronization.

## Guide + Patterns

### Pattern 1: NumPy Preprocess -> Torch Inference -> NumPy Postprocess

This is the simplest handoff. Keep image decode / resize / normalize in NumPy,
run the model in Torch, then hand the outputs back to NumPy.

```python
from ml_pipes import Decode, Normalize, Pipeline, Resize
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

Use this when the model is Torch-native but the rest of the pipeline is already
clear and standardized in NumPy operators.

### Pattern 2: NumPy Preprocess -> Torch Inference + Torch Postprocess -> NumPy Output

This is the right shape when the postprocess is tensor-heavy and benefits from
staying in Torch.

The canonical example in this repo is
[`examples/torch/run_mask2former_torch_postprocess.py`](examples/torch/run_mask2former_torch_postprocess.py),
which keeps mask resizing, query scoring, filtering, winner assignment, and
mask-to-box conversion in Torch before converting back to NumPy segmentations.

Key shape:

```python
from ml_pipes import Pipeline, ToSegmentations
from ml_pipes.torch import (
    ToNumpyRegistry,
    TorchMasksToBoxes,
    TorchResizeMasks,
    TorchSigmoid,
    TorchSoftmax,
)

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
> Keep postprocess in Torch when the needed operators already exist and you want
> to stay on-device for the heavy tensor work.

### Pattern 3: Torch Registry Handoff Around A Specific Step

Sometimes only one step is better in Torch. In that case, convert the registry,
run the Torch step, and hand it back.

```python
from ml_pipes import Extract, Infer, Pipeline
from ml_pipes.torch import ToNumpyRegistry, ToTorchRegistry, TorchNMS

pipeline = Pipeline([
    Infer("detector.onnx"),
    Extract("boxes", "scores", "classes"),
    ToTorchRegistry(device="cpu"),
    TorchNMS(),
    ToNumpyRegistry(),
])
```

This keeps the rest of the pipeline unchanged while letting one postprocess
operator run in Torch.

### When To Keep Postprocess In Torch vs Hand Off To NumPy

Keep postprocess in Torch when:

- the operators already exist under `ml_pipes.torch`
- the work is tensor-heavy enough that staying on-device matters
- the pipeline reads more clearly as a Torch-native tensor flow

Hand off to NumPy when:

- the remaining steps are already implemented and tested in NumPy
- you need projection, visualization, or other NumPy-oriented utilities
- readability matters more than squeezing out one more Torch-only stage

> [!TIP]
> Hand off to NumPy when downstream steps are simpler, more mature, or already
> standardized there. Do not keep a pipeline in Torch just for symmetry.

The paired Mask2Former examples are the best concrete reference:

- [`examples/torch/run_mask2former_torch_postprocess.py`](examples/torch/run_mask2former_torch_postprocess.py)
- [`examples/torch/run_mask2former_numpy_postprocess.py`](examples/torch/run_mask2former_numpy_postprocess.py)

## Quick Reference

This page is not a full operator catalog. Use it for orientation, and use the
package exports plus tests as the detailed source of truth.

Torch operator groups:

- conversion / boundary:
  `ToTorch`, `ToTorchRegistry`, `ToNumpy`, `ToNumpyRegistry`, `ToDevice`
- execution / runtime:
  `TorchInfer`, `TorchExtract`, `TorchCollate`, `TorchDistribute`, `TorchAsType`
- tensor math / indexing / filtering:
  `TorchArgMax`, `TorchGatherRows`, `TorchGatherScores`, `TorchTopK`,
  `TorchTopKIndices2D`, `TorchSlice`, `TorchSoftmax`, `TorchSigmoid`,
  `TorchMultiplyTensors`, `TorchCreateTensorMask`, `TorchCreateTensorMaskByThreshold`,
  `TorchApplyTensorMask`, `TorchSelectTensors`, `TorchFilterTensorsByScore`,
  `TorchFilterTensorsByMasksArea`, `TorchSortTensorsBy`
- segmentation helpers:
  `TorchWeightMasksByScores`, `TorchResizeMasks`, `TorchMeanMaskScores`,
  `TorchMasksToBoxes`, `TorchNMS`

Look here for the current exported surface:

- [`src/ml_pipes/torch/__init__.py`](src/ml_pipes/torch/__init__.py)

Look here for behavioral edge cases and usage patterns:

- [`tests/test_torch.py`](tests/test_torch.py)

Use [`OPERATORS.md`](OPERATORS.md) for the general operator model and
composition style. Use this page for Torch-specific runtime and boundary rules.

## Rationale

Torch support is isolated under `ml_pipes.torch` so that:

- core `ml_pipes` stays lightweight and NumPy-oriented
- Torch remains an optional dependency
- domain crossings stay visible in the pipeline list

NumPy and Torch both remain first-class because they serve different strengths:

- NumPy is a natural fit for generic preprocessing, projection, visualization,
  and framework-agnostic pipelines
- Torch is a natural fit for model execution and Torch-native tensor postprocess

That split keeps the pipeline readable. You can choose Torch where it helps
without making the whole library Torch-only.
