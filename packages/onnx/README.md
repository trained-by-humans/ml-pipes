# ml-pipes-onnx

ONNX runtime operators and output types for `ml-pipes`. This package owns only
the ONNX runtime surface. Tensor operators used in the same pipeline still
come from `ml_pipes.tensor`.

## Package Reference

| Field          | Value                                                                              |
|----------------|------------------------------------------------------------------------------------|
| Package        | `ml-pipes-onnx`                                                                    |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `onnxruntime`                         |
| Public modules | `ml_pipes.onnx`                                                                    |
| Content        | ONNX runtime invocation; runtime outputs; output extraction and batch distribution |

See [`docs/PACKAGES.md`](../../docs/PACKAGES.md) for direct package installs,
umbrella profiles, and the full package matrix.
