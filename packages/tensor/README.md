# ml-pipes-tensor

`ml-pipes-tensor` is a package for the `ml-pipes` framework. It adds tensor
payloads and tensor-domain operators to `ml-pipes`.

While its operators are plain callables, this package is best used as part of
an `ml-pipes` pipeline. See the main project docs and examples for the broader
framework usage model.

## Package Reference

| Field          | Value                                                                                                                                                   |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-tensor`                                                                                                                                       |
| Depends on     | `ml-pipes-core`, `numpy`                                                                                                                                |
| Public modules | `ml_pipes.tensor`                                                                                                                                       |
| Content        | tensor payloads and registries; dtype casting; shape and layout transforms; tensor selection, filtering, ranking, and masking; collation and arithmetic |

See
[`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
