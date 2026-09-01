# ml-pipes-torch Index

This page catalogs the Torch package surface in `ml_pipes.torch`.
For package overview, scope, design principles, and usage patterns, see
[`README.md`](./README.md).

The package mirrors `ml_pipes.tensor` for shared tensor postprocess,
and `ml_pipes.onnx` for runtime staging shape. Use those package indexes for
the NumPy-side variants and typed vision outputs.
For the Tensor and ONNX operator coverage matrix, see
[`COVERAGE.md`](./COVERAGE.md).

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md).
For the cross-package package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Package Primitives

| Surface | Notes |
|---|---|
| `TensorPayload` | One-tensor Torch boundary type. |
| `TensorRegistry` | Multi-tensor Torch working set used by most postprocess operators in this package. |
| `RuntimeOutputs` | Value type used between Torch runtime invocation and output extraction or distribution. |

## Domain Boundaries And Device Movement

| Operator | Input -> Output | Notes |
|---|---|---|
| `ToTorch(device="cpu", dtype=None, copy=False)` | `tensor.TensorPayload` -> `torch.TensorPayload` | Converts one NumPy tensor payload into the Torch domain. |
| `ToNumpy(dtype=None, copy=False)` | `torch.TensorPayload` -> `tensor.TensorPayload` | Converts one Torch tensor payload back into the NumPy domain. |
| `ToTorchRegistry(device="cpu", dtype=None, copy=False)` | `tensor.TensorRegistry` -> `torch.TensorRegistry` | Converts a tensor registry into the Torch domain. |
| `ToNumpyRegistry(dtype=None, copy=False)` | `torch.TensorRegistry` -> `tensor.TensorRegistry` | Converts a Torch registry back into the NumPy domain. |
| `ToDevice(device)` | Torch values -> same Torch values | Moves Torch-backed payloads, registries, runtime outputs, or Torch sequences to another device. |
| `SynchronizeTensors()` | Torch values -> same Torch values | Forces synchronization at a chosen Torch boundary. |

## Model Runtime And Output Handling

| Operator | Input -> Output | Notes |
|---|---|---|
| `Infer(model, input_name=None, input_layout="NCHW", ...)` | `TensorPayload` -> `RuntimeOutputs` | Runs a Torch module on one input payload. For the NumPy-side runtime mirror, see `ml_pipes.onnx`. |
| `Extract(*names, as_=...)` | `RuntimeOutputs` -> `TensorRegistry` | Extracts named Torch outputs into a Torch registry. |
| `Distribute()` | `RuntimeOutputs` -> `list[RuntimeOutputs]` | Splits a batched runtime output back into per-sample runtime outputs. |

## Generic Tensor Operators

### Public Aliases

| Public alias | Primary name | Note |
|---|---|---|
| `GatherScores(...)` | `GatherRows(...)` | `scores` comes from the common use of gathering one selected score from each row. |
| `BinarizeTensor(...)` | `CreateTensorMask(...)` | `binarize` comes from older wording for turning a tensor into a boolean mask. |
| `BinarizeTensorByThreshold(...)` | `CreateTensorMaskByThreshold(...)` | `binarize` comes from threshold-based boolean mask creation. |

### Dtype And Shape

| Operator | Notes |
|---|---|
| `AsType(dtype, src=None, as_=None)` | Casts a raw Torch tensor value or a named registry tensor to a new dtype. |
| `Squeeze(src, axis=None, as_=None)` | Removes unit dimensions from a named tensor. |
| `Transpose(src, axes=None, as_=None)` | Permutes tensor axes. |
| `Slice(src, at, as_=None)` | Slices a named tensor and stores the result back into the registry. |
| `Scale(src, by, as_=None)` | Multiplies a tensor by a scalar or per-column factors. |

### Ranking, Selection, And Masking

| Operator | Notes |
|---|---|
| `GatherRows(src, indices, as_=None)` | Row-wise gather driven by another registry tensor of indices. |
| `TopK(src, k, values_as, indices_as)` | Returns the top-k values and indices from a 1D tensor. |
| `TopKIndices2D(src, k, ...)` | Returns top-k values plus row and column indices from a 2D tensor. |
| `ArgMax(src, axis=-1, as_=None)` | Computes argmax along an axis. |
| `CreateTensorMask(src, predicate, as_)` | Builds a boolean mask from a tensor. |
| `CreateTensorMaskByThreshold(src, threshold, as_=None)` | Convenience threshold-based mask creation. |
| `ApplyTensorMask(*srcs, mask, as_=...)` | Applies one boolean mask across one or more tensors. |
| `SelectTensors(*srcs, indices, as_=...)` | Applies integer-index selection across one or more tensors. |
| `FilterTensors(*srcs, by, predicate, as_=...)` | Filters one or more tensors by a predicate on another tensor. |
| `SortTensorsBy(*srcs, by, descending=True, as_=...)` | Reorders one or more tensors by a ranking tensor. |

### Math And Mapping

| Operator | Notes |
|---|---|
| `Softmax(src, axis=-1, as_=None)` | Softmax over a named tensor. |
| `Sigmoid(src, as_=None)` | Element-wise sigmoid. |
| `MultiplyTensors(left, right, as_=None)` | Element-wise multiplication of two named tensors. |
| `MapTensor(src, fn, as_=None)` | Applies an arbitrary tensor-to-tensor mapping function. |

### Batch Assembly

| Operator | Input -> Output | Notes |
|---|---|---|
| `Collate()` | `list[TensorPayload]` -> `TensorPayload` | Stacks or concatenates Torch payloads into one batched payload. |
