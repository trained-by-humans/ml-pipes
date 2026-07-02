# ml-pipes-onnx Operator Index

This file catalogs the ONNX runtime operators shipped with `ml-pipes-onnx`
through `ml_pipes.onnx`.

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md). For the cross-package
package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## Runtime And Output Handling

| Operator | Input -> Output | Notes |
|---|---|---|
| `Infer(model_path, providers=..., input_name=..., input_layout=..., dtype=..., output_layouts=..., serialize=False)` | `TensorPayload` -> `RuntimeOutputs` | Runs ONNX Runtime and validates input layout and dtype expectations before inference. |
| `Extract(*names, as_=...)` | `RuntimeOutputs` -> `TensorRegistry` | Extracts named ONNX outputs into a tensor registry. |
| `Distribute()` | `RuntimeOutputs` -> `list[RuntimeOutputs]` | Splits a batched runtime output back into per-sample runtime outputs. |

The package also owns the `RuntimeOutputs` value type used between ONNX
runtime invocation and registry extraction.
