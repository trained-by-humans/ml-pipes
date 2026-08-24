# ml-pipes-core

`ml-pipes-core` is the main package of the `ml-pipes` framework. It carries
the shared execution harness, generic operators, and framework tooling that
the other packages build on.

`ml-pipes` uses a multi-package layout so this core can stay small and
domain-agnostic, while preprocessing, runtime integration, and task-specific
postprocess live in their own optional packages.

If you are starting with `ml-pipes`, begin with the main project
[`README.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/README.md)
for the framework overview, installation model, and end-to-end examples.

## Package Reference

| Field             | Value                                                                                                                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package           | `ml-pipes-core`                                                                                                                                                                         |
| Reference         | [`docs/INDEX.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/packages/core/docs/INDEX.md)                                                                                 |
| Optional installs | `inspection`, `otel`                                                                                                                                                                   |
| Depends on        | `—`                                                                                                                                                                                    |
| Public modules    | `ml_pipes.core`, `ml_pipes.standard`, `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, `ml_pipes.inspection`                |
| Content           | pipeline composition; generic flow-control and data operators; validation, tracing, inspection, benchmarking, factory, and CLI tooling                                                 |

The `inspection` optional install adds the shared inspection renderer
dependencies only. Package-owned inspection formatting comes from the package
modules you also install and import.

See
[`docs/PACKAGES.md`](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md)
for direct package installs, umbrella profiles, and the full package matrix.
