# ml-pipes-torch Index

This page catalogs the Torch package surface in `ml_pipes.torch`.
For package overview, scope, design principles, and usage patterns, see
[`README.md`](./README.md).

Internally, the package is organized into boundary, runtime, tensor, and
vision-helper modules while the public import surface remains
`ml_pipes.torch`.
The package mirrors `ml_pipes.tensor` for shared tensor postprocess,
`ml_pipes.onnx` for runtime staging shape, and `ml_pipes.vision` for
Torch-native vision postprocess operators. Use those package indexes for the NumPy-side
variants and for typed vision outputs.

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md).
For the cross-package package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Package Primitives

| Surface | Notes |
|---|---|
| `TorchTensorPayload` | One-tensor Torch boundary type. |
| `TorchTensorRegistry` | Multi-tensor Torch working set used by most postprocess operators in this package. |
| `TorchRuntimeOutputs` | Value type used between Torch runtime invocation and output extraction or distribution. |

## Public Aliases

| Public alias | Primary name | Note |
|---|---|---|
| `TorchGatherScores(...)` | `TorchGatherRows(...)` | `scores` comes from the common use of gathering one selected score from each row. |
| `TorchBinarizeTensor(...)` | `TorchCreateTensorMask(...)` | `binarize` comes from older wording for turning a tensor into a boolean mask. |
| `TorchBinarizeTensorByThreshold(...)` | `TorchCreateTensorMaskByThreshold(...)` | `binarize` comes from threshold-based boolean mask creation. |

## Domain Boundaries And Device Movement

| Operator | Input -> Output | Notes |
|---|---|---|
| `ToTorch(device="cpu", dtype=None, copy=False)` | `TensorPayload` -> `TorchTensorPayload` | Converts one NumPy tensor payload into the Torch domain. |
| `ToNumpy(dtype=None, copy=False)` | `TorchTensorPayload` -> `TensorPayload` | Converts one Torch tensor payload back into the NumPy domain. |
| `ToTorchRegistry(device="cpu", dtype=None, copy=False)` | `TensorRegistry` -> `TorchTensorRegistry` | Converts a tensor registry into the Torch domain. |
| `ToNumpyRegistry(dtype=None, copy=False)` | `TorchTensorRegistry` -> `TensorRegistry` | Converts a Torch registry back into the NumPy domain. |
| `ToDevice(device)` | Torch values -> same Torch values | Moves Torch-backed payloads, registries, runtime outputs, or Torch sequences to another device. |
| `TorchSynchronizeTensors()` | Torch values -> same Torch values | Forces synchronization at a chosen Torch boundary. |

## Model Runtime And Output Handling

| Operator | Input -> Output | Notes |
|---|---|---|
| `TorchInfer(model, input_name=None, input_layout="NCHW", ...)` | `TorchTensorPayload` -> `TorchRuntimeOutputs` | Runs a Torch module on one input payload. For the NumPy-side runtime mirror, see `ml_pipes.onnx`. |
| `TorchExtract(*names, as_=...)` | `TorchRuntimeOutputs` -> `TorchTensorRegistry` | Extracts named Torch outputs into a Torch registry. |
| `TorchDistribute()` | `TorchRuntimeOutputs` -> `list[TorchRuntimeOutputs]` | Splits a batched runtime output back into per-sample runtime outputs. |

## Generic Tensor Operators

### Dtype And Shape

| Operator | Notes |
|---|---|
| `TorchAsType(dtype, src=None, as_=None)` | Casts a raw Torch tensor value or a named registry tensor to a new dtype. |
| `TorchSqueeze(src, axis=None, as_=None)` | Removes unit dimensions from a named tensor. |
| `TorchTranspose(src, axes=None, as_=None)` | Permutes tensor axes. |
| `TorchSlice(src, at, as_=None)` | Slices a named tensor and stores the result back into the registry. |
| `TorchScale(src, by, as_=None)` | Multiplies a tensor by a scalar or per-column factors. |

### Ranking, Selection, And Masking

| Operator | Notes |
|---|---|
| `TorchGatherRows(src, indices, as_=None)` | Row-wise gather driven by another registry tensor of indices. |
| `TorchTopK(src, k, values_as, indices_as)` | Returns the top-k values and indices from a 1D tensor. |
| `TorchTopKIndices2D(src, k, ...)` | Returns top-k values plus row and column indices from a 2D tensor. |
| `TorchArgMax(src, axis=-1, as_=None)` | Computes argmax along an axis. |
| `TorchCreateTensorMask(src, predicate, as_)` | Builds a boolean mask from a tensor. |
| `TorchCreateTensorMaskByThreshold(src, threshold, as_=None)` | Convenience threshold-based mask creation. |
| `TorchApplyTensorMask(*srcs, mask, as_=...)` | Applies one boolean mask across one or more tensors. |
| `TorchSelectTensors(*srcs, indices, as_=...)` | Applies integer-index selection across one or more tensors. |
| `TorchFilterTensors(*srcs, by, predicate, as_=...)` | Filters one or more tensors by a predicate on another tensor. |
| `TorchSortTensorsBy(*srcs, by, descending=True, as_=...)` | Reorders one or more tensors by a ranking tensor. |

### Math And Mapping

| Operator | Notes |
|---|---|
| `TorchSoftmax(src, axis=-1, as_=None)` | Softmax over a named tensor. |
| `TorchSigmoid(src, as_=None)` | Element-wise sigmoid. |
| `TorchMultiplyTensors(left, right, as_=None)` | Element-wise multiplication of two named tensors. |
| `TorchMapTensor(src, fn, as_=None)` | Applies an arbitrary tensor-to-tensor mapping function. |

### Batch Assembly

| Operator | Input -> Output | Notes |
|---|---|---|
| `TorchCollate()` | `list[TorchTensorPayload]` -> `TorchTensorPayload` | Stacks or concatenates Torch payloads into one batched payload. |

## Vision Postprocess Operators

These mirror NumPy-side postprocess operators in `ml_pipes.vision` so tensor
values can remain on-device. For image payloads, source-image projection, and
NumPy-side rendering, see the vision package index.

| Operator | Notes |
|---|---|
| `TorchConvertBoxFormat(src="boxes", from_=..., to="xyxy", as_=None)` | Converts between `xyxy`, `xywh`, and `cxcywh` box formats. |
| `TorchNMS(...)` | Confidence filtering plus per-class non-maximum suppression on registry tensors. |
| `TorchFilterTensorsByScore(...)` | Filters one or more tensors by a score threshold. |
| `TorchFilterTensorsByClasses(...)` | Filters one or more tensors by allowed class ids. |
| `TorchFilterTensorsByMasksArea(...)` | Filters one or more tensors by mask area. |
| `TorchWeightMasksByScores(...)` | Weights masks by per-instance scores. |
| `TorchResizeMasks(...)` | Resizes instance masks to an image shape. |
| `TorchMeanMaskScores(...)` | Computes one score per instance from dense mask scores over foreground masks. |
| `TorchMasksToBoxes(...)` | Derives boxes from binary masks. |
| `TorchReconstructMasks(coefficients, prototypes, as_)` | Reconstructs instance masks from coefficients and prototypes. |
