# ml-pipes-torch

`ml-pipes-torch` is a package for the `ml-pipes` framework. It adds the Torch
execution domain to `ml-pipes`.

While its operators are plain callables, this package is best used as part of
an `ml-pipes` pipeline. See the main project docs and examples for the broader
framework usage model.

## Package Reference

| Field          | Value                                                                                                                                                                   |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-torch`                                                                                                                                                        |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `torch`, `torchvision`                                                                                                     |
| Public modules | `ml_pipes.torch`                                                                                                                                                        |
| Content        | Torch payloads and registries; NumPy and Torch conversion; device and dtype movement; Torch inference; tensor filtering, ranking, and masking; mask postprocess helpers |

See
[`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
