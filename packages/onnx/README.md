# ml-pipes-onnx

`ml-pipes-onnx` is a package for the `ml-pipes` framework. It adds the ONNX
execution domain to `ml-pipes`.

While its operators are plain callables, this package is best used as part of
an `ml-pipes` pipeline. See the main project docs and examples for the broader
framework usage model.

## Package Reference

| Field          | Value                                                                              |
|----------------|------------------------------------------------------------------------------------|
| Package        | `ml-pipes-onnx`                                                                    |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `onnxruntime`                         |
| Public modules | `ml_pipes.onnx`                                                                    |
| Content        | ONNX runtime invocation; runtime outputs; output extraction and batch distribution |

See
[`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
