# ml-pipes-tensor

Tensor payloads and tensor registry operators for `ml-pipes`. This package
owns the tensor operators even when a pipeline also uses ONNX, vision, or
Torch packages.

## Package Reference

| Field          | Value                                                                                                                                                   |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-tensor`                                                                                                                                       |
| Depends on     | `ml-pipes-core`, `numpy`                                                                                                                                |
| Public modules | `ml_pipes.tensor`                                                                                                                                       |
| Content        | tensor payloads and registries; dtype casting; shape and layout transforms; tensor selection, filtering, ranking, and masking; collation and arithmetic |

See [`docs/PACKAGES.md`](../../docs/PACKAGES.md) for direct package installs,
umbrella profiles, and the full package matrix.
