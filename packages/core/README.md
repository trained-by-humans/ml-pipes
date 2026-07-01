# ml-pipes-core

Core runtime, validation, tracing, inspection, CLI, and standard operators
for `ml-pipes`. This package owns the shared framework harness and the
standard operator family. It also exposes optional installs for features that
only add dependencies to modules that already live here.

## Package Reference

| Field             | Value                                                                                                                                                                                |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Package           | `ml-pipes-core`                                                                                                                                                                      |
| Optional installs | `inspection`, `otel`                                                                                                                                                                 |
| Depends on        | `numpy`, `typing_extensions`                                                                                                                                                         |
| Public modules    | `ml_pipes.core`, `ml_pipes.standard`, `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, `ml_pipes.inspection`              |
| Content           | pipeline composition; context and control; standard routing, regions, batching, scatter, and data ops; validation; tracing and collectors; inspection; benchmarking; factory and CLI |

See [`docs/PACKAGES.md`](../../docs/PACKAGES.md) for direct package installs,
umbrella profiles, and the full package matrix.
