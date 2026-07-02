# ml-pipes-vision

`ml-pipes-vision` is a package for the `ml-pipes` framework. It adds the
vision domain to `ml-pipes`.

While its operators are plain callables, this package is best used as part of
an `ml-pipes` pipeline. See the main project docs and examples for the broader
framework usage model.

## Package Reference

| Field          | Value                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-vision`                                                                                              |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `opencv-python`                                                   |
| Public modules | `ml_pipes.vision`                                                                                              |
| Content        | image payloads and file I/O; image transforms; detection; segmentation; density; tiling; rendering and logging |

See
[`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
