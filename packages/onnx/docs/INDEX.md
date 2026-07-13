# ml-pipes-onnx Index

This page catalogs the ONNX package surface in `ml_pipes.onnx`.
For package overview, scope, design principles, and usage patterns, see
[`README.md`](./README.md).

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md).
For the cross-package package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Package Primitives

| Surface          | Notes                                                                                  |
|------------------|----------------------------------------------------------------------------------------|
| `RuntimeOutputs` | Value type used between ONNX runtime invocation and output extraction or distribution. |

## Runtime And Output Handling

| Operator                                                                                                             | Input -> Output                            | Notes                                                                                                      |
|----------------------------------------------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `Infer(model_path, providers=..., input_name=..., input_layout=..., dtype=..., output_layouts=..., serialize=False)` | `TensorPayload` -> `RuntimeOutputs`        | Runs ONNX Runtime on one input payload and validates input layout and dtype expectations before inference. |
| `Extract(*names, as_=...)`                                                                                           | `RuntimeOutputs` -> `TensorRegistry`       | Extracts named ONNX outputs into a tensor registry.                                                        |
| `Distribute()`                                                                                                       | `RuntimeOutputs` -> `list[RuntimeOutputs]` | Splits a batched runtime output back into per-sample runtime outputs.                                      |
