# Torch Operator Coverage

This matrix shows which `ml_pipes.tensor` and `ml_pipes.vision` operators can
currently remain in the Torch domain through an equivalent
`ml_pipes.torch` operator.

`ml_pipes.tensor` has a complete generic-operator mirror. `ml_pipes.vision`
has a deliberately narrow on-device postprocess mirror; image preparation,
projection, tiling, rendering, logging, and density remain Vision-owned
NumPy-side stages.

An equivalent name here identifies the current Torch-side operator. It does
not by itself guarantee behavioral conformance between NumPy and Torch
implementations; the conformance requirements are tracked by [issue
#70](https://github.com/trained-by-humans/ml-pipes/issues/70).

| Target package | Operator | Equivalent Torch operator |
|---|---|---|
| `ml_pipes.tensor` | `AsType` | `TorchAsType` |
| `ml_pipes.tensor` | `Squeeze` | `TorchSqueeze` |
| `ml_pipes.tensor` | `Transpose` | `TorchTranspose` |
| `ml_pipes.tensor` | `Slice` | `TorchSlice` |
| `ml_pipes.tensor` | `Scale` | `TorchScale` |
| `ml_pipes.tensor` | `GatherRows` | `TorchGatherRows` |
| `ml_pipes.tensor` | `TopK` | `TorchTopK` |
| `ml_pipes.tensor` | `TopKIndices2D` | `TorchTopKIndices2D` |
| `ml_pipes.tensor` | `ArgMax` | `TorchArgMax` |
| `ml_pipes.tensor` | `CreateTensorMask` | `TorchCreateTensorMask` |
| `ml_pipes.tensor` | `CreateTensorMaskByThreshold` | `TorchCreateTensorMaskByThreshold` |
| `ml_pipes.tensor` | `ApplyTensorMask` | `TorchApplyTensorMask` |
| `ml_pipes.tensor` | `SelectTensors` | `TorchSelectTensors` |
| `ml_pipes.tensor` | `FilterTensors` | `TorchFilterTensors` |
| `ml_pipes.tensor` | `SortTensorsBy` | `TorchSortTensorsBy` |
| `ml_pipes.tensor` | `Softmax` | `TorchSoftmax` |
| `ml_pipes.tensor` | `Sigmoid` | `TorchSigmoid` |
| `ml_pipes.tensor` | `MultiplyTensors` | `TorchMultiplyTensors` |
| `ml_pipes.tensor` | `MapTensor` | `TorchMapTensor` |
| `ml_pipes.tensor` | `Collate` | `TorchCollate` |
| `ml_pipes.vision` | `LoadFile` | — |
| `ml_pipes.vision` | `Decode` | — |
| `ml_pipes.vision` | `Resize` | — |
| `ml_pipes.vision` | `ConvertColorSpace` | — |
| `ml_pipes.vision` | `Normalize` | — |
| `ml_pipes.vision` | `ConvertBoxFormat` | `TorchConvertBoxFormat` |
| `ml_pipes.vision` | `NMS` | `TorchNMS` |
| `ml_pipes.vision` | `NMM` | — |
| `ml_pipes.vision` | `FilterTensorsByScore` | `TorchFilterTensorsByScore` |
| `ml_pipes.vision` | `FilterTensorsByClasses` | `TorchFilterTensorsByClasses` |
| `ml_pipes.vision` | `FilterTensorsByBoxArea` | — |
| `ml_pipes.vision` | `FilterTensorsByMasksArea` | `TorchFilterTensorsByMasksArea` |
| `ml_pipes.vision` | `ProjectBoxes` | — |
| `ml_pipes.vision` | `ReconstructMasks` | `TorchReconstructMasks` |
| `ml_pipes.vision` | `ProjectMasks` | — |
| `ml_pipes.vision` | `ProjectRoIMasks` | — |
| `ml_pipes.vision` | `ResizeMasks` | `TorchResizeMasks` |
| `ml_pipes.vision` | `MasksToBoxes` | `TorchMasksToBoxes` |
| `ml_pipes.vision` | `WeightMasksByScores` | `TorchWeightMasksByScores` |
| `ml_pipes.vision` | `MeanMaskScores` | `TorchMeanMaskScores` |
| `ml_pipes.vision` | `Tile` | — |
| `ml_pipes.vision` | `Stitch` | — |
| `ml_pipes.vision` | `BlendImages` | — |
| `ml_pipes.vision` | `DrawBoxes` | — |
| `ml_pipes.vision` | `DrawDensityOverlay` | — |
| `ml_pipes.vision` | `DrawMasks` | — |
| `ml_pipes.vision` | `SaveImage` | — |
| `ml_pipes.vision` | `LogDetections` | — |
| `ml_pipes.vision` | `ClampDensity` | — |
| `ml_pipes.vision` | `SumDensity` | — |
| `ml_pipes.vision` | `ProjectDensityMap` | — |
| `ml_pipes.vision` | `DensityToHeatmap` | — |

## Public Aliases

The following Tensor aliases also have direct Torch aliases:

| Target package | Operator | Equivalent Torch operator |
|---|---|---|
| `ml_pipes.tensor` | `GatherScores` | `TorchGatherScores` |
| `ml_pipes.tensor` | `BinarizeTensor` | `TorchBinarizeTensor` |
| `ml_pipes.tensor` | `BinarizeTensorByThreshold` | `TorchBinarizeTensorByThreshold` |

## Not Covered By This Matrix

Torch-specific boundary and device operators (`ToTorch`, `ToNumpy`,
`ToTorchRegistry`, `ToNumpyRegistry`, `ToDevice`, and
`TorchSynchronizeTensors`) have no Tensor or Vision counterpart because they
make entry to, exit from, and movement within the Torch domain explicit.

Torch runtime operators (`TorchInfer`, `TorchExtract`, and
`TorchDistribute`) parallel the ONNX runtime staging shape, not a Tensor or
Vision operator.
