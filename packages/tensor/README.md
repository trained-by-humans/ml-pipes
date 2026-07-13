# ml-pipes-tensor

`ml-pipes-tensor` is the shared tensor workbench for `ml-pipes`.

Use this package when a pipeline needs NumPy-side tensor shaping, ranking,
masking, filtering, collation, or light arithmetic between a runtime boundary
and a typed task result. In practice, it often carries the reusable tensor
postprocess that sits after ONNX or Torch output extraction and before a task
package such as vision finalizes the result.

Operators can be used as plain callables, but this package is best used inside
an `ml-pipes` pipeline.

## Package Reference

| Field          | Value                                                                                                            |
|----------------|------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-tensor`                                                                                                |
| Guide          | [`docs/README.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/tensor/docs/README.md)       |
| Reference      | [`docs/INDEX.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/tensor/docs/INDEX.md)         |
| Depends on     | `ml-pipes-core`                                                                                                  |
| Public modules | `ml_pipes.tensor`                                                                                                |
| Content        | shared tensor values; NumPy-side tensor shaping and selection; masking, ranking, collation, and light arithmetic |

See [`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
