# ml-pipes-onnx

`ml-pipes-onnx` brings ONNX Runtime inference into `ml-pipes`.

Use this package when a model is already exported to ONNX and you want the
runtime call to appear as one explicit pipeline stage. The package is
intentionally thin: it handles the runtime boundary and the values
immediately around that boundary, then hands model outputs to packages such as
`ml-pipes-tensor` or `ml-pipes-vision` for shared postprocess.

Operators can be used as plain callables, but this package is best used inside
an `ml-pipes` pipeline.

## Package Reference

| Field          | Value                                                                                                           |
|----------------|-----------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-onnx`                                                                                                |
| Guide          | [`docs/README.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/onnx/docs/README.md)      |
| Reference      | [`docs/INDEX.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/onnx/docs/INDEX.md)        |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`                                                                              |
| Public modules | `ml_pipes.onnx`                                                                                                 |
| Content        | ONNX Runtime integration; runtime-output handling; output extraction and batch distribution                     |

See [`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
