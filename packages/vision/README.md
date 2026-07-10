# ml-pipes-vision

`ml-pipes-vision` adds image-oriented pipeline stages and typed vision outputs
to `ml-pipes`.

Use this package for the parts of a pipeline that start with image inputs,
prepare them for inference, and turn model results back into user-facing
vision predictions. Today it is strongest around image preprocessing,
detection, segmentation, density estimation, tiling, and
visualization/logging helpers.

Operators can be used as plain callables, but this package is best used inside
an `ml-pipes` pipeline.

## Package Reference

| Field          | Value                                                                                                               |
|----------------|---------------------------------------------------------------------------------------------------------------------|
| Package        | `ml-pipes-vision` ([Index](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/vision/docs/INDEX.md))  |
| Depends on     | `ml-pipes-core`, `ml-pipes-tensor`                                                                                  |
| Public modules | `ml_pipes.vision`                                                                                                   |
| Content        | image input handling; preprocessing; typed vision predictions; tiling; rendering and logging                        |

See [`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
