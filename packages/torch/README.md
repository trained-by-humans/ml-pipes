# ml-pipes-torch

`ml-pipes-torch` brings Torch-backed execution into `ml-pipes`.

Use this package when a pipeline needs to cross into Torch for model
inference, device-aware execution, or postprocess that is worth keeping on the
Torch side. In practice, it often sits between NumPy-oriented preparation or
postprocess stages so mixed pipelines can move in and out of the Torch domain
explicitly.

Operators can be used as plain callables, but this package is best used inside
an `ml-pipes` pipeline.

## Package Reference

| Field          | Value                                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-torch`                                                                                          |
| Guide          | [`docs/README.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/torch/docs/README.md) |
| Reference      | [`docs/INDEX.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/torch/docs/INDEX.md)   |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`                                                                        |
| Public modules | `ml_pipes.torch`                                                                                          |
| Content        | Torch execution stages; NumPy/Torch boundary crossing; device placement; Torch-native tensor postprocess  |

See [`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
