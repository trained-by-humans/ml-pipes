# Torch Operator Coverage

`ml_pipes.torch` mirrors the generic tensor postprocess surface of
`ml_pipes.tensor` and the runtime staging shape of `ml_pipes.onnx`. This page
shows the coverage of those mirrors and identifies the corresponding public
operators.

> [!CAUTION]
> The same operator name indicates the same intended pipeline role; it does
> not promise behavioral conformance. NumPy and Torch can differ in dtype
> handling, numerical results, device behavior, and edge cases.

## Tensor Package Coverage

| `ml_pipes.tensor` | `ml_pipes.torch` | Note |
|---|---|---|
| `AsType` | `AsType` | |
| `Squeeze` | `Squeeze` | |
| `Transpose` | `Transpose` | |
| `Slice` | `Slice` | |
| `Scale` | `Scale` | |
| `GatherRows` | `GatherRows` | |
| `GatherScores` | `GatherScores` | Public alias of `GatherRows`. |
| `TopK` | `TopK` | |
| `TopKIndices2D` | `TopKIndices2D` | |
| `ArgMax` | `ArgMax` | |
| `CreateTensorMask` | `CreateTensorMask` | |
| `CreateTensorMaskByThreshold` | `CreateTensorMaskByThreshold` | |
| `BinarizeTensor` | `BinarizeTensor` | Public alias of `CreateTensorMask`. |
| `BinarizeTensorByThreshold` | `BinarizeTensorByThreshold` | Public alias of `CreateTensorMaskByThreshold`. |
| `ApplyTensorMask` | `ApplyTensorMask` | |
| `SelectTensors` | `SelectTensors` | |
| `FilterTensors` | `FilterTensors` | |
| `SortTensorsBy` | `SortTensorsBy` | |
| `Softmax` | `Softmax` | |
| `Sigmoid` | `Sigmoid` | |
| `MultiplyTensors` | `MultiplyTensors` | |
| `MapTensor` | `MapTensor` | |
| `Collate` | `Collate` | |

For operator details, see the [Tensor package index](../../tensor/docs/INDEX.md).

## ONNX Package Coverage

| `ml_pipes.onnx` | `ml_pipes.torch` | Note |
|---|---|---|
| `Infer` | `Infer` | Torch `Infer` accepts a `torch.nn.Module`; it is not interchangeable with ONNX Runtime model loading or configuration. |
| `Extract` | `Extract` | |
| `Distribute` | `Distribute` | |

For operator details, see the [ONNX package index](../../onnx/docs/INDEX.md).
