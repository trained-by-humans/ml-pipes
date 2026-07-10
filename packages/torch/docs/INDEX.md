# ml-pipes-torch Index

This page catalogs the Torch package surface in `ml_pipes.torch`.

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md).
For the cross-package package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Package Primitives

| Surface | Notes |
|---|---|
| `TorchTensorPayload` | Main Torch-backed single-tensor boundary. |
| `TorchTensorRegistry` | Main named working set for Torch-side postprocess. |
| `TorchRuntimeOutputs` | Runtime value used between `TorchInfer` and output extraction or distribution. |

## Public Aliases

| Public alias | Primary name | Note |
|---|---|---|
| `TorchGatherScores(...)` | `TorchGatherRows(...)` | `scores` comes from the common use of gathering one selected score from each row. |
| `TorchBinarizeTensor(...)` | `TorchCreateTensorMask(...)` | `binarize` comes from older wording for turning a tensor into a boolean mask. |
| `TorchBinarizeTensorByThreshold(...)` | `TorchCreateTensorMaskByThreshold(...)` | `binarize` comes from threshold-based boolean mask creation. |

## Domain Boundaries And Device Movement

| Operator | Notes |
|---|---|
| `ToTorch(device="cpu", dtype=None, copy=False)` | Converts one `TensorPayload` into `TorchTensorPayload`. |
| `ToNumpy(dtype=None, copy=False)` | Converts one `TorchTensorPayload` back into `TensorPayload`. |
| `ToTorchRegistry(device="cpu", dtype=None, copy=False)` | Converts a `TensorRegistry` into `TorchTensorRegistry`. |
| `ToNumpyRegistry(dtype=None, copy=False)` | Converts a `TorchTensorRegistry` back into `TensorRegistry`. |
| `ToDevice(device)` | Moves Torch-backed payloads, registries, or runtime outputs to another device. |
| `TorchSynchronizeTensors()` | Forces device synchronization at a chosen pipeline boundary. |

## Runtime And Registry Setup

| Operator | Notes |
|---|---|
| `TorchAsType(dtype, src=None, as_=None)` | Casts Torch-backed tensor values or named registry tensors. |
| `TorchInfer(model, input_layout="NCHW", ...)` | Runs a Torch-native callable on one `TorchTensorPayload` and returns `TorchRuntimeOutputs`. |
| `TorchExtract(*names, as_=...)` | Extracts named Torch outputs into `TorchTensorRegistry`. |
| `TorchDistribute()` | Splits batched `TorchRuntimeOutputs` into per-sample outputs. |
| `TorchCollate()` | Stacks `list[TorchTensorPayload]` into one batched payload. |

## Registry Math, Ranking, And Masking

| Operator | Notes |
|---|---|
| `TorchArgMax(...)` | Argmax over a named Torch tensor. |
| `TorchGatherRows(...)` | Row-wise gather driven by another registry tensor of indices. |
| `TorchTopK(...)` / `TorchTopKIndices2D(...)` | Top-k ranking helpers for 1D and 2D tensors. |
| `TorchSlice(...)` | Slices a named tensor in a Torch registry. |
| `TorchSoftmax(...)` / `TorchSigmoid(...)` | Standard per-tensor nonlinearities. |
| `TorchMultiplyTensors(...)` | Element-wise multiplication of two named tensors. |
| `TorchCreateTensorMask(...)` | Builds boolean masks from Torch tensors. |
| `TorchCreateTensorMaskByThreshold(...)` | Threshold-based mask creation. |
| `TorchApplyTensorMask(...)` | Applies one boolean mask across one or more tensors. |
| `TorchSelectTensors(...)` | Applies integer-index selection across one or more tensors. |
| `TorchFilterTensorsByScore(...)` | Filters tensors by a score threshold. |
| `TorchFilterTensorsByClasses(...)` | Filters tensors by allowed class ids. |
| `TorchFilterTensorsByMasksArea(...)` | Filters tensors by mask area. |
| `TorchSortTensorsBy(...)` | Reorders tensors by a ranking tensor. |

## Segmentation Helpers

| Operator | Notes |
|---|---|
| `TorchWeightMasksByScores(...)` | Weights masks by per-instance scores. |
| `TorchResizeMasks(...)` | Resizes masks while staying in the Torch domain. |
| `TorchMeanMaskScores(...)` | Computes mean scores over masks or masked areas. |
| `TorchMasksToBoxes(...)` | Derives boxes from masks. |
| `TorchNMS(...)` | Torch-native non-maximum suppression for registry tensors. |
