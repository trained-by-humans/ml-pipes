# ml-pipes-torch

Torch tensor payloads and torch-domain operators for `ml-pipes`. This package
owns the Torch-domain surface; vision operators remain in
`ml_pipes.vision`.

## Package Reference

| Field          | Value                                                                                                                                                                   |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-torch`                                                                                                                                                        |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `torch`, `torchvision`                                                                                                     |
| Public modules | `ml_pipes.torch`                                                                                                                                                        |
| Content        | Torch payloads and registries; NumPy and Torch conversion; device and dtype movement; Torch inference; tensor filtering, ranking, and masking; mask postprocess helpers |

See [`docs/PACKAGES.md`](../../docs/PACKAGES.md) for direct package installs,
umbrella profiles, and the full package matrix.
